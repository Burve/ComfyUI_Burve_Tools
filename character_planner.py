import copy
import json
import os


PRESET_FILE = os.path.join(os.path.dirname(__file__), "character_planner_presets.json")
FACE_LOCK_SYSTEM_INSTRUCTION = (
    "If a face reference image is provided, preserve that person's exact facial "
    "features and identity in the generated character."
)
CHARACTER_RACE_PIPE_KIND = "burve.character_race_pipe"
CHARACTER_RACE_PIPE_VERSION = 1
RACE_TRAIT_KEYS = (
    "ears",
    "horns",
    "wings",
    "tail",
    "legs_feet",
    "skin_surface",
    "head_features",
    "hands_arms",
    "extra_notes",
)
GENDER_AVATAR_TYPES = {
    "female": "adult_female_photorealistic",
    "male": "adult_male_photorealistic",
}
GENDER_IGNORED_CONTROLS = {
    "female": ["male_chest_cm", "male_body_frame_preset", "male_outfit_variant"],
    "male": ["bust_cm", "underbust_cm", "body_frame_preset", "outfit_variant"],
}
_PRESET_CACHE = None


def clamp(value, lower, upper):
    return max(lower, min(upper, float(value)))


def load_character_planner_presets():
    global _PRESET_CACHE

    if _PRESET_CACHE is None:
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            _PRESET_CACHE = json.load(f)

    return copy.deepcopy(_PRESET_CACHE)


def strip_json_wrappers(raw_text):
    if raw_text is None:
        return ""

    cleaned = str(raw_text).strip()
    if not cleaned:
        return ""

    if cleaned[:6].lower() == "<json>" and cleaned[-7:].lower() == "</json>":
        return cleaned[6:-7].strip()

    return cleaned


def parse_plan_overrides_json(raw_text):
    cleaned = strip_json_wrappers(raw_text)
    if not cleaned:
        return {}, False

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid plan_overrides_json: {e.msg}") from e

    if not isinstance(parsed, dict):
        raise ValueError("plan_overrides_json must be a JSON object.")

    return parsed, True


def normalize_custom_text_override(raw_text):
    if raw_text is None:
        return None

    cleaned = str(raw_text).strip()
    if not cleaned:
        return None

    return cleaned


def derive_age_group(age_years):
    age_years = int(age_years)
    if 18 <= age_years <= 24:
        return "young_adult"
    if 25 <= age_years <= 39:
        return "adult"
    if 40 <= age_years <= 59:
        return "mature_adult"
    return "older_adult"


def resolve_custom_text_overrides(ui_values):
    resolved_ui_values = copy.deepcopy(ui_values)
    applied = []

    custom_hair_color = normalize_custom_text_override(ui_values.get("custom_hair_color"))
    if custom_hair_color is not None:
        resolved_ui_values["hair_color"] = custom_hair_color
        applied.append("hair_color")

    custom_skin_tone = normalize_custom_text_override(ui_values.get("custom_skin_tone"))
    if custom_skin_tone is not None:
        resolved_ui_values["skin_tone"] = custom_skin_tone
        applied.append("skin_tone")

    custom_race = normalize_custom_text_override(ui_values.get("custom_race"))
    if custom_race is not None:
        resolved_ui_values["race"] = custom_race
        applied.append("race")

    return resolved_ui_values, applied


def normalize_race_pipe(race_override_pipe):
    if race_override_pipe is None:
        return None

    if not isinstance(race_override_pipe, dict):
        raise ValueError("Invalid race_override_pipe: expected a dict payload.")

    if race_override_pipe.get("kind") != CHARACTER_RACE_PIPE_KIND:
        raise ValueError("Invalid race_override_pipe: unexpected kind.")

    if race_override_pipe.get("version") != CHARACTER_RACE_PIPE_VERSION:
        raise ValueError("Invalid race_override_pipe: unsupported version.")

    race_name = race_override_pipe.get("race_name", "")
    if not isinstance(race_name, str):
        raise ValueError("Invalid race_override_pipe: race_name must be a string.")

    traits = race_override_pipe.get("traits", {})
    if not isinstance(traits, dict):
        raise ValueError("Invalid race_override_pipe: traits must be an object.")

    normalized_traits = {}
    for key in RACE_TRAIT_KEYS:
        value = traits.get(key, "")
        if not isinstance(value, str):
            raise ValueError(f"Invalid race_override_pipe: traits.{key} must be a string.")
        normalized_traits[key] = value

    summary = race_override_pipe.get("summary", "")
    if not isinstance(summary, str):
        raise ValueError("Invalid race_override_pipe: summary must be a string.")

    return {
        "kind": CHARACTER_RACE_PIPE_KIND,
        "version": CHARACTER_RACE_PIPE_VERSION,
        "race_name": race_name,
        "traits": normalized_traits,
        "summary": summary,
    }


def resolve_race_pipe(race_override_pipe):
    normalized_pipe = normalize_race_pipe(race_override_pipe)
    if normalized_pipe is None:
        return None, {}

    patch = {}
    if normalize_custom_text_override(normalized_pipe["race_name"]) is not None:
        patch.setdefault("identity", {}).setdefault("race", {})["base"] = normalized_pipe["race_name"].strip()

    fantasy_traits_patch = {}
    for key in RACE_TRAIT_KEYS:
        value = normalize_custom_text_override(normalized_pipe["traits"].get(key, ""))
        if value is not None:
            fantasy_traits_patch[key] = value

    if fantasy_traits_patch:
        patch["fantasy_traits"] = fantasy_traits_patch

    return normalized_pipe, patch


def deep_merge(base, patch):
    merged = copy.deepcopy(base)
    _deep_merge_in_place(merged, patch)
    return merged


def _deep_merge_in_place(target, patch):
    if not isinstance(patch, dict):
        return patch

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_in_place(target[key], value)
        else:
            target[key] = copy.deepcopy(value)

    return target


def filter_to_schema(value, schema):
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return {}
        return {
            key: filter_to_schema(value[key], schema[key])
            for key in schema
            if key in value
        }

    if isinstance(schema, list):
        if isinstance(value, list):
            return copy.deepcopy(value)
        return []

    return copy.deepcopy(value)


def find_unknown_paths(value, schema, prefix=""):
    unknown_paths = []

    if not isinstance(value, dict):
        return unknown_paths

    if not isinstance(schema, dict):
        if prefix:
            unknown_paths.append(prefix)
        return unknown_paths

    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            unknown_paths.append(path)
            continue
        unknown_paths.extend(find_unknown_paths(child, schema[key], path))

    return unknown_paths


def _get_path(data, path, default=None):
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _build_numeric_validation_errors(value, schema, prefix=""):
    errors = []

    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return [f"{prefix or 'plan'} must be an object."]
        for key, schema_value in schema.items():
            if key not in value:
                continue
            path = f"{prefix}.{key}" if prefix else key
            errors.extend(_build_numeric_validation_errors(value[key], schema_value, path))
        return errors

    if isinstance(schema, list):
        if value is None or isinstance(value, list):
            return []
        return [f"{prefix} must be a list."]

    if isinstance(schema, bool):
        if isinstance(value, bool):
            return []
        return [f"{prefix} must be a boolean."]

    if isinstance(schema, (int, float)) and not isinstance(schema, bool):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [f"{prefix} must be numeric."]
        if isinstance(schema, float) and 0.0 <= schema <= 1.0 and not 0.0 <= float(value) <= 1.0:
            return [f"{prefix} must stay within 0..1."]
        return []

    if isinstance(schema, str):
        if isinstance(value, str):
            return []
        return [f"{prefix} must be a string."]

    return []


def _compute_derived_patch(ui_values):
    height_cm = int(ui_values["height_cm"])
    weight_kg = int(ui_values["weight_kg"])
    bust_cm = int(ui_values["bust_cm"])
    underbust_cm = int(ui_values["underbust_cm"])
    waist_cm = int(ui_values["waist_cm"])
    full_hip_cm = int(ui_values["full_hip_cm"])
    musculature_tone = float(ui_values["musculature_tone"])
    body_fat = float(ui_values["body_fat"])

    cup_delta = max(bust_cm - underbust_cm, 0)
    bmi = weight_kg / ((height_cm / 100.0) ** 2) if height_cm > 0 else 0.0
    waist_to_hip_ratio = waist_cm / max(full_hip_cm, 1)
    waist_to_bust_ratio = waist_cm / max(bust_cm, 1)

    volume = clamp(0.3 + (cup_delta / 45.0), 0.24, 0.86)
    projection = clamp(0.28 + (cup_delta / 50.0), 0.22, 0.82)
    base_width = clamp(0.4 + ((bust_cm - underbust_cm) / 120.0), 0.34, 0.68)

    waist_definition = clamp(1.42 - (waist_to_hip_ratio * 1.08), 0.36, 0.86)
    rib_to_waist_taper = clamp(1.15 - waist_to_bust_ratio, 0.35, 0.8)
    waist_to_hip_transition = clamp(1.32 - waist_to_hip_ratio, 0.4, 0.86)

    abdomen_flatness = clamp(0.82 - (body_fat * 0.28) + (musculature_tone * 0.12), 0.38, 0.86)
    lower_abdomen_softness = clamp(0.08 + (body_fat * 0.48), 0.08, 0.55)
    oblique_visibility = clamp(0.06 + (musculature_tone * 0.36) - (body_fat * 0.14), 0.02, 0.48)

    high_hip_cm = int(round((waist_cm * 0.45) + (full_hip_cm * 0.55)))
    gluteal_projection_cm = int(round(clamp(4.0 + ((full_hip_cm - waist_cm) / 4.5), 4.0, 12.0)))

    bmi_influence = clamp((bmi - 19.0) / 10.0, 0.0, 1.0)
    firmness = clamp(0.76 - (body_fat * 0.24), 0.42, 0.82)
    ptosis = clamp(0.08 + (body_fat * 0.18) + (volume * 0.12), 0.08, 0.36)
    separation = clamp(0.34 - (cup_delta / 150.0), 0.22, 0.36)

    return {
        "body": {
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "measurements": {
                "bust_cm": bust_cm,
                "underbust_cm": underbust_cm,
                "waist_cm": waist_cm,
                "high_hip_cm": high_hip_cm,
                "full_hip_cm": full_hip_cm,
                "gluteal_projection_cm": gluteal_projection_cm,
            },
            "musculature": {
                "overall_tone": musculature_tone,
                "upper_body_definition": clamp((musculature_tone * 0.78), 0.0, 1.0),
                "core_definition": clamp((musculature_tone * 0.9), 0.0, 1.0),
                "lower_body_definition": clamp((musculature_tone * 1.05), 0.0, 1.0),
            },
            "soft_tissue": {
                "overall_body_fat": body_fat,
            },
        },
        "chest": {
            "size": {
                "volume": volume,
                "projection": projection,
                "base_width": base_width,
            },
            "placement": {
                "separation": separation,
            },
            "support_characteristics": {
                "firmness": firmness,
                "ptosis": ptosis,
            },
        },
        "waist_and_torso": {
            "waist_definition": waist_definition,
            "rib_to_waist_taper": rib_to_waist_taper,
            "waist_to_hip_transition": waist_to_hip_transition,
            "abdomen": {
                "flatness": abdomen_flatness,
                "lower_abdomen_softness": lower_abdomen_softness,
                "oblique_visibility": oblique_visibility,
            },
        },
        "hips_glutes": {
            "glutes": {
                "projection": clamp(0.48 + (bmi_influence * 0.1) + ((full_hip_cm - waist_cm) / 180.0), 0.48, 0.74),
            }
        },
        "symmetry_and_variation": {
            "natural_variation": clamp(0.08 + (body_fat * 0.14), 0.08, 0.2),
        },
    }


def _compute_male_derived_patch(ui_values):
    height_cm = int(ui_values["height_cm"])
    weight_kg = int(ui_values["weight_kg"])
    male_chest_cm = int(ui_values["male_chest_cm"])
    waist_cm = int(ui_values["waist_cm"])
    full_hip_cm = int(ui_values["full_hip_cm"])
    musculature_tone = float(ui_values["musculature_tone"])
    body_fat = float(ui_values["body_fat"])

    bmi = weight_kg / ((height_cm / 100.0) ** 2) if height_cm > 0 else 0.0
    waist_to_hip_ratio = waist_cm / max(full_hip_cm, 1)
    high_hip_cm = int(round((waist_cm * 0.48) + (full_hip_cm * 0.52)))
    gluteal_projection_cm = int(round(clamp(3.5 + ((full_hip_cm - waist_cm) / 5.0), 3.0, 11.0)))
    abdomen_flatness = clamp(0.84 - (body_fat * 0.24) + (musculature_tone * 0.16), 0.42, 0.9)
    lower_abdomen_softness = clamp(0.06 + (body_fat * 0.38), 0.06, 0.44)
    oblique_visibility = clamp(0.1 + (musculature_tone * 0.42) - (body_fat * 0.18), 0.04, 0.6)
    waist_definition = clamp(1.18 - (waist_to_hip_ratio * 0.84), 0.28, 0.74)
    rib_to_waist_taper = clamp(1.08 - (waist_cm / max(male_chest_cm, 1)), 0.24, 0.78)
    waist_to_hip_transition = clamp(1.16 - waist_to_hip_ratio, 0.28, 0.7)
    bmi_influence = clamp((bmi - 20.0) / 11.0, 0.0, 1.0)

    return {
        "body": {
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "measurements": {
                "waist_cm": waist_cm,
                "high_hip_cm": high_hip_cm,
                "full_hip_cm": full_hip_cm,
                "gluteal_projection_cm": gluteal_projection_cm,
            },
            "musculature": {
                "overall_tone": musculature_tone,
                "upper_body_definition": clamp((musculature_tone * 0.95), 0.0, 1.0),
                "core_definition": clamp((musculature_tone * 0.88), 0.0, 1.0),
                "lower_body_definition": clamp((musculature_tone * 0.9), 0.0, 1.0),
            },
            "soft_tissue": {
                "overall_body_fat": body_fat,
            },
        },
        "male_torso": {
            "measurements": {
                "chest_cm": male_chest_cm,
            },
            "shape": {
                "pectoral_mass": clamp(0.32 + ((male_chest_cm - waist_cm) / 85.0), 0.28, 0.82),
                "pectoral_definition": clamp(0.18 + (musculature_tone * 0.62) - (body_fat * 0.18), 0.08, 0.86),
                "v_taper": clamp(0.24 + ((male_chest_cm - waist_cm) / 120.0), 0.18, 0.8),
                "lat_width": clamp(
                    0.20 + (musculature_tone * 0.40) + ((male_chest_cm - waist_cm) / 200.0),
                    0.18,
                    0.78,
                ),
            },
        },
        "waist_and_torso": {
            "waist_definition": waist_definition,
            "rib_to_waist_taper": rib_to_waist_taper,
            "waist_to_hip_transition": waist_to_hip_transition,
            "abdomen": {
                "flatness": abdomen_flatness,
                "lower_abdomen_softness": lower_abdomen_softness,
                "oblique_visibility": oblique_visibility,
            },
        },
        "hips_glutes": {
            "glutes": {
                "projection": clamp(0.42 + (bmi_influence * 0.08) + ((full_hip_cm - waist_cm) / 220.0), 0.42, 0.66),
            }
        },
        "symmetry_and_variation": {
            "natural_variation": clamp(0.06 + (body_fat * 0.12), 0.06, 0.18),
        },
    }


def _selected_preset_maps(gender, preset_data):
    if gender == "male":
        return preset_data["male_body_frame_presets"], preset_data["male_outfit_variants"]
    return preset_data["body_frame_presets"], preset_data["outfit_variants"]


def build_curated_ui_patch(ui_values, preset_data):
    gender = ui_values["gender"]
    frame_presets, outfit_variants = _selected_preset_maps(gender, preset_data)
    body_frame_preset = ui_values["male_body_frame_preset"] if gender == "male" else ui_values["body_frame_preset"]
    outfit_variant = ui_values["male_outfit_variant"] if gender == "male" else ui_values["outfit_variant"]
    outfit_color = ui_values["outfit_color"]

    if body_frame_preset not in frame_presets:
        raise ValueError(f"Unsupported body_frame_preset: {body_frame_preset}")
    if outfit_variant not in outfit_variants:
        raise ValueError(f"Unsupported outfit_variant: {outfit_variant}")

    frame_patch = copy.deepcopy(frame_presets[body_frame_preset])
    outfit_patch = copy.deepcopy(outfit_variants[outfit_variant])
    outfit_patch["color"] = outfit_color

    ui_patch = {
        "avatar_type": GENDER_AVATAR_TYPES[gender],
        "identity": {
            "gender": gender,
            "age_years": int(ui_values["age_years"]),
            "age_group": derive_age_group(ui_values["age_years"]),
            "race": {
                "base": ui_values["race"],
            },
            "face_reference": {
                "enabled": bool(ui_values["use_face_reference"]),
                "source_type": "image",
                "image_slot": "face_reference_input",
                "strength": float(ui_values["face_reference_strength"]),
                "fallback_if_missing": "disabled",
            },
            "skin_tone": {
                "base": ui_values["skin_tone"],
                "undertone": ui_values["undertone"],
            },
        },
        "rendering": {
            "pose": ui_values["pose"],
        },
        "base_outfit": outfit_patch,
        "hair": {
            "scalp_hair": {
                "enabled": True,
                "length": ui_values["hair_length"],
                "color": ui_values["hair_color"],
            }
        },
    }

    ui_patch = deep_merge(frame_patch, ui_patch)
    derived_patch = _compute_male_derived_patch(ui_values) if gender == "male" else _compute_derived_patch(ui_values)
    ui_patch = deep_merge(ui_patch, derived_patch)
    return ui_patch


def _positive_number_paths_for_gender(gender):
    base_paths = [
        ("body", "height_cm"),
        ("body", "weight_kg"),
        ("body", "measurements", "waist_cm"),
        ("body", "measurements", "high_hip_cm"),
        ("body", "measurements", "full_hip_cm"),
        ("rendering", "camera", "lens_mm"),
    ]

    if gender == "male":
        return base_paths + [
            ("male_torso", "measurements", "chest_cm"),
        ]

    return base_paths + [
        ("body", "measurements", "bust_cm"),
        ("body", "measurements", "underbust_cm"),
    ]


def summarize_fantasy_traits(plan):
    fantasy_traits = _get_path(plan, ("fantasy_traits",), {}) or {}
    active_traits = []

    for key in RACE_TRAIT_KEYS:
        value = fantasy_traits.get(key, "")
        if key == "extra_notes":
            if normalize_custom_text_override(value) is not None:
                active_traits.append("extra_notes")
            continue

        if value not in ("", "none", None):
            active_traits.append(f"{key}={value}")

    return ", ".join(active_traits) if active_traits else "none"


def validate_character_plan(plan, schema, ui_values, preset_data, face_reference_present=False):
    errors = []

    requested_gender = ui_values["gender"]
    plan_gender = _get_path(plan, ("identity", "gender"))
    if plan_gender not in GENDER_AVATAR_TYPES:
        errors.append("identity.gender must be either 'female' or 'male'.")
    elif plan_gender != requested_gender:
        errors.append("plan_overrides_json contradicts gender.")

    expected_avatar_type = GENDER_AVATAR_TYPES.get(plan_gender)
    if expected_avatar_type and plan.get("avatar_type") != expected_avatar_type:
        errors.append(f"avatar_type must match identity.gender ({expected_avatar_type}).")

    age_years = _get_path(plan, ("identity", "age_years"))
    if not isinstance(age_years, int) or isinstance(age_years, bool) or not 18 <= age_years <= 80:
        errors.append("identity.age_years must be an integer between 18 and 80.")
    else:
        expected_age_group = derive_age_group(age_years)
        if _get_path(plan, ("identity", "age_group")) != expected_age_group:
            errors.append("identity.age_group must match identity.age_years.")

    for path in _positive_number_paths_for_gender(plan_gender or requested_gender):
        value = _get_path(plan, path)
        joined = ".".join(path)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
            errors.append(f"{joined} must be positive.")

    errors.extend(_build_numeric_validation_errors(plan, schema))

    if plan_gender == "female":
        if _get_path(plan, ("male_torso",), None) not in (None, {}):
            errors.append("female plans cannot include male_torso.")
        if _get_path(plan, ("base_outfit", "variant")) not in preset_data["outfit_variants"]:
            errors.append("female plans cannot use male-only outfit variants.")
    elif plan_gender == "male":
        if _get_path(plan, ("chest",), None) not in (None, {}):
            errors.append("male plans cannot include female chest.")
        if _get_path(plan, ("base_outfit", "variant")) not in preset_data["male_outfit_variants"]:
            errors.append("male plans cannot use female-only outfit variants.")
        if _get_path(plan, ("base_outfit", "top", "type")) != "none":
            errors.append("male plans must keep base_outfit.top.type set to 'none'.")

    requested_face_lock = bool(ui_values["use_face_reference"])
    merged_face_config = _get_path(plan, ("identity", "face_reference"), {}) or {}
    merged_face_enabled = bool(merged_face_config.get("enabled"))

    if requested_face_lock != merged_face_enabled:
        errors.append("plan_overrides_json contradicts use_face_reference.")

    if merged_face_enabled:
        if not face_reference_present:
            errors.append("use_face_reference is enabled but no face_reference_image is connected.")
        if merged_face_config.get("source_type") != "image":
            errors.append("identity.face_reference.source_type must remain 'image'.")
        if not merged_face_config.get("image_slot"):
            errors.append("identity.face_reference.image_slot must be set when face reference is enabled.")

    if errors:
        raise ValueError(errors[0])


def _build_prompt_sections(plan, schema):
    identity_section = filter_to_schema(plan.get("identity", {}), schema.get("identity", {}))
    face_section = identity_section.pop("face_reference", {})

    body_section = filter_to_schema(plan.get("body", {}), schema.get("body", {}))
    measurements_section = {
        "body_measurements": body_section.pop("measurements", {}),
    }

    chest_section = filter_to_schema(plan.get("chest", {}), schema.get("chest", {}))
    if chest_section:
        measurements_section["chest"] = chest_section

    male_torso_section = filter_to_schema(plan.get("male_torso", {}), schema.get("male_torso", {}))
    if male_torso_section:
        measurements_section["male_torso"] = male_torso_section

    return [
        (
            "Identity",
            {
                "version": plan.get("version"),
                "avatar_type": plan.get("avatar_type"),
                "identity": identity_section,
            },
        ),
        (
            "Race / Fantasy Traits",
            {
                "fantasy_traits": filter_to_schema(plan.get("fantasy_traits", {}), schema.get("fantasy_traits", {})),
            },
        ),
        (
            "Face Handling",
            {
                "face_reference": face_section,
            },
        ),
        (
            "Rendering",
            {
                "rendering": filter_to_schema(plan.get("rendering", {}), schema.get("rendering", {})),
            },
        ),
        (
            "Body / Frame / Proportions",
            {
                "body": body_section,
                "symmetry_and_variation": filter_to_schema(
                    plan.get("symmetry_and_variation", {}),
                    schema.get("symmetry_and_variation", {}),
                ),
            },
        ),
        (
            "Measurements",
            measurements_section,
        ),
        (
            "Torso / Hips / Legs",
            {
                "waist_and_torso": filter_to_schema(
                    plan.get("waist_and_torso", {}),
                    schema.get("waist_and_torso", {}),
                ),
                "shoulders_arms_hands": filter_to_schema(
                    plan.get("shoulders_arms_hands", {}),
                    schema.get("shoulders_arms_hands", {}),
                ),
                "hips_glutes": filter_to_schema(
                    plan.get("hips_glutes", {}),
                    schema.get("hips_glutes", {}),
                ),
                "legs": filter_to_schema(plan.get("legs", {}), schema.get("legs", {})),
                "feet": filter_to_schema(plan.get("feet", {}), schema.get("feet", {})),
            },
        ),
        (
            "Skin / Hair",
            {
                "skin": filter_to_schema(plan.get("skin", {}), schema.get("skin", {})),
                "hair": filter_to_schema(plan.get("hair", {}), schema.get("hair", {})),
            },
        ),
        (
            "Base Outfit",
            {
                "base_outfit": filter_to_schema(plan.get("base_outfit", {}), schema.get("base_outfit", {})),
            },
        ),
        (
            "Constraints",
            {
                "constraints": filter_to_schema(plan.get("constraints", {}), schema.get("constraints", {})),
            },
        ),
    ]


def compile_character_prompt(plan, schema):
    sections = _build_prompt_sections(plan, schema)

    lines = [
        "Create a reusable adult photorealistic base-character reference image intended for later outfit swaps and body-shape consistency.",
        "Keep the result full-body, anatomically plausible, neutral in presentation, and strictly non-explicit.",
        "Use minimal contour-preserving basewear rather than compressive sportswear so the torso, hips, legs, and overall silhouette remain readable for future styling.",
        "Follow this structured character plan exactly:",
    ]

    for title, section_data in sections:
        lines.append("")
        lines.append(f"{title}:")
        lines.append(json.dumps(section_data, indent=2, ensure_ascii=True))

    return "\n".join(lines).strip()


def resolve_reference_manifest(use_face_reference, face_reference_present, extra_reference_batch_sizes, limit=14):
    reference_labels = []
    warnings = []

    if use_face_reference and face_reference_present:
        reference_labels.append("face_reference")
    elif face_reference_present and not use_face_reference:
        warnings.append("Dedicated face reference image was provided but ignored because use_face_reference is disabled.")

    for batch_index, batch_size in enumerate(extra_reference_batch_sizes or []):
        if batch_size is None:
            continue
        batch_size = int(batch_size)
        if batch_size < 0:
            raise ValueError("Reference batch sizes must be non-negative.")
        for frame_index in range(batch_size):
            reference_labels.append(f"extra_reference_images[{batch_index}]#{frame_index}")

    total_available = len(reference_labels)
    truncated = total_available > limit
    if truncated:
        warnings.append(
            f"Reference images truncated from {total_available} to {limit} to fit the Gemini 14-image limit."
        )

    return {
        "reference_labels": reference_labels[:limit],
        "count": min(total_available, limit),
        "total_available": total_available,
        "truncated": truncated,
        "warnings": warnings,
    }


def build_summary(
    face_lock_active,
    gender,
    age_years,
    age_group,
    resolved_race,
    fantasy_traits_label,
    ignored_gender_specific_controls,
    outfit_variant,
    reference_manifest,
    overrides_applied,
    custom_text_overrides_applied,
    warnings,
):
    custom_overrides_label = ", ".join(custom_text_overrides_applied) if custom_text_overrides_applied else "none"
    ignored_controls_label = (
        ", ".join(ignored_gender_specific_controls) if ignored_gender_specific_controls else "none"
    )
    lines = [
        f"Gender: {gender}",
        f"Age: {age_years} ({age_group})",
        f"Race: {resolved_race}",
        f"Fantasy traits: {fantasy_traits_label}",
        f"Ignored gender-specific controls: {ignored_controls_label}",
        f"Face lock: {'active' if face_lock_active else 'inactive'}",
        f"Outfit variant: {outfit_variant}",
        f"Reference images: {reference_manifest['count']} / 14",
        f"Overrides applied: {'yes' if overrides_applied else 'no'}",
        f"Custom text overrides: {custom_overrides_label}",
    ]

    if warnings:
        lines.append(f"Warnings: {'; '.join(warnings)}")
    else:
        lines.append("Warnings: none")

    return "\n".join(lines)


def build_character_plan(
    ui_values,
    plan_overrides_json="",
    face_reference_present=False,
    extra_reference_batch_sizes=None,
    race_override_pipe=None,
):
    preset_data = load_character_planner_presets()
    requested_gender = ui_values["gender"]
    default_plan_key = "default_plan_male" if requested_gender == "male" else "default_plan"
    default_plan = preset_data[default_plan_key]
    resolved_ui_values, custom_text_overrides_applied = resolve_custom_text_overrides(ui_values)

    plan = copy.deepcopy(default_plan)
    plan = deep_merge(plan, build_curated_ui_patch(resolved_ui_values, preset_data))

    normalized_race_pipe, race_pipe_patch = resolve_race_pipe(race_override_pipe)
    if race_pipe_patch:
        plan = deep_merge(plan, race_pipe_patch)

    override_patch, overrides_applied = parse_plan_overrides_json(plan_overrides_json)
    if overrides_applied:
        plan = deep_merge(plan, override_patch)

    validate_character_plan(
        plan=plan,
        schema=default_plan,
        ui_values=ui_values,
        preset_data=preset_data,
        face_reference_present=face_reference_present,
    )

    unknown_paths = find_unknown_paths(plan, default_plan)
    reference_manifest = resolve_reference_manifest(
        use_face_reference=bool(_get_path(plan, ("identity", "face_reference", "enabled"), False)),
        face_reference_present=face_reference_present,
        extra_reference_batch_sizes=extra_reference_batch_sizes or [],
        limit=14,
    )

    warnings = list(reference_manifest["warnings"])
    if unknown_paths:
        warnings.append(
            "Unsupported override keys not mapped by the prompt compiler: " + ", ".join(unknown_paths)
        )

    face_lock_active = bool(_get_path(plan, ("identity", "face_reference", "enabled"), False)) and face_reference_present
    prompt = compile_character_prompt(plan, default_plan)
    character_plan_json = json.dumps(plan, indent=2, ensure_ascii=True)
    system_instructions = FACE_LOCK_SYSTEM_INSTRUCTION if face_lock_active else ""
    gender = _get_path(plan, ("identity", "gender"), requested_gender)
    age_years = _get_path(plan, ("identity", "age_years"))
    age_group = _get_path(plan, ("identity", "age_group"))
    resolved_race = _get_path(plan, ("identity", "race", "base"), "human")
    outfit_variant = _get_path(plan, ("base_outfit", "variant"), "custom")
    fantasy_traits_label = summarize_fantasy_traits(plan)
    ignored_gender_specific_controls = GENDER_IGNORED_CONTROLS.get(gender, [])
    summary = build_summary(
        face_lock_active=face_lock_active,
        gender=gender,
        age_years=age_years,
        age_group=age_group,
        resolved_race=resolved_race,
        fantasy_traits_label=fantasy_traits_label,
        ignored_gender_specific_controls=ignored_gender_specific_controls,
        outfit_variant=outfit_variant,
        reference_manifest=reference_manifest,
        overrides_applied=overrides_applied,
        custom_text_overrides_applied=custom_text_overrides_applied,
        warnings=warnings,
    )

    return {
        "plan": plan,
        "character_plan_json": character_plan_json,
        "prompt": prompt,
        "system_instructions": system_instructions,
        "summary": summary,
        "warnings": warnings,
        "unknown_paths": unknown_paths,
        "reference_manifest": reference_manifest,
        "overrides_applied": overrides_applied,
        "face_lock_active": face_lock_active,
        "custom_text_overrides_applied": custom_text_overrides_applied,
        "fantasy_traits_label": fantasy_traits_label,
        "ignored_gender_specific_controls": ignored_gender_specific_controls,
        "resolved_race": resolved_race,
        "normalized_race_pipe": normalized_race_pipe,
    }
