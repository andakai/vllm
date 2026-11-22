// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::{ApiError, invalid_request};
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WeightCheckerAction {
    Checksum,
    Reset,
    Compare,
}

#[derive(Debug, Deserialize)]
pub(crate) struct WeightCheckerRequest {
    action: WeightCheckerAction,
    #[serde(default)]
    baseline: Option<BTreeMap<String, String>>,
    #[serde(default)]
    names: Option<Vec<String>>,
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum WeightCheckerResponse {
    Checksums {
        checksums: BTreeMap<String, String>,
    },
    Reset {
        status: &'static str,
    },
    Compare {
        #[serde(rename = "match")]
        matched: bool,
        mismatches: Vec<String>,
    },
}

/// The requested parameter names that own a checksum key.
///
/// Keys are `{rank prefix}:{tensor}`, where the tensor is a parameter name,
/// optionally followed by `.{partition}` for parameters stored in parts.
fn requested_names<'a>(key: &str, names: &'a BTreeSet<String>) -> Vec<&'a String> {
    let tensor = key.splitn(5, ':').last().unwrap_or(key);
    tensor
        .match_indices('.')
        .map(|(end, _)| &tensor[..end])
        .chain([tensor])
        .filter_map(|prefix| names.get(prefix))
        .collect()
}

/// Checksum, reset, or compare model weights against a baseline.
///
/// `names` limits checksum and compare to those parameters.
pub async fn weight_checker(
    State(state): State<Arc<AppState>>,
    body: Result<Json<WeightCheckerRequest>, JsonRejection>,
) -> Result<Json<WeightCheckerResponse>, ApiError> {
    let Json(body) = body?;
    let client = state.engine_core_client();
    let no_args = || {
        (
            Vec::<JsonValue>::new(),
            BTreeMap::<String, JsonValue>::new(),
        )
    };

    if body
        .names
        .as_ref()
        .is_some_and(|names| body.action == WeightCheckerAction::Reset || names.is_empty())
    {
        return Err(invalid_request!(
            param = Some("names"),
            "'names' must be a non-empty list, with action 'checksum' or 'compare'"
        ));
    }
    if body.action == WeightCheckerAction::Reset {
        let (args, kwargs) = no_args();
        client
            .collective_rpc("reset_weights", None, args, kwargs)
            .await
            .map_err(|error| utility_call_error("reset_weights", error))?;
        return Ok(Json(WeightCheckerResponse::Reset { status: "reset" }));
    }
    if body.action == WeightCheckerAction::Compare && body.baseline.is_none() {
        return Err(invalid_request!(
            param = Some("baseline"),
            "action='compare' requires a 'baseline' object"
        ));
    }

    let (args, mut kwargs) = no_args();
    if let Some(names) = &body.names {
        kwargs.insert("names".to_string(), JsonValue::from(names.clone()));
    }
    let workers = client
        .collective_rpc("compute_weight_checksums", None, args, kwargs)
        .await
        .map_err(|error| utility_call_error("compute_weight_checksums", error))?;
    let mut checksums = BTreeMap::new();
    for worker in workers {
        let worker: BTreeMap<String, String> = rmpv::ext::from_value(worker).map_err(|error| {
            ApiError::server_error(format!("invalid weight checksums: {error}"))
        })?;
        for (key, digest) in worker {
            if checksums.contains_key(&key) {
                return Err(ApiError::server_error(format!(
                    "Duplicate weight checksum keys: {key}"
                )));
            }
            checksums.insert(key, digest);
        }
    }

    let mut baseline = body.baseline;
    if let Some(names) = body.names {
        let requested: BTreeSet<String> = names.into_iter().collect();
        let found: BTreeSet<&String> = checksums
            .keys()
            .flat_map(|key| requested_names(key, &requested))
            .collect();
        let unknown: Vec<&String> = requested.iter().filter(|n| !found.contains(n)).collect();
        if !unknown.is_empty() {
            return Err(invalid_request!(
                param = Some("names"),
                "Unknown parameter names: {unknown:?}"
            ));
        }
        if let Some(baseline) = baseline.as_mut() {
            baseline.retain(|key, _| !requested_names(key, &requested).is_empty());
        }
    }
    let Some(baseline) = baseline.filter(|_| body.action == WeightCheckerAction::Compare) else {
        return Ok(Json(WeightCheckerResponse::Checksums { checksums }));
    };
    let mismatches: Vec<String> = checksums
        .keys()
        .chain(baseline.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|key| checksums.get(*key) != baseline.get(*key))
        .cloned()
        .collect();
    Ok(Json(WeightCheckerResponse::Compare {
        matched: mismatches.is_empty(),
        mismatches,
    }))
}
