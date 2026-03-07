import json
import unittest
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from character_planner import build_character_plan, resolve_reference_manifest


def make_ui_values(**overrides):
    values = {
        "height_cm": 170,
        "weight_kg": 58,
        "bust_cm": 90,
        "underbust_cm": 74,
        "waist_cm": 64,
        "full_hip_cm": 95,
        "body_frame_preset": "balanced",
        "skin_tone": "light_medium",
        "undertone": "neutral",
        "hair_color": "dark_blonde",
        "hair_length": "long",
        "musculature_tone": 0.35,
        "body_fat": 0.28,
        "pose": "neutral_a_pose",
        "outfit_variant": "classic_triangle",
        "outfit_color": "neutral_gray",
        "use_face_reference": False,
        "face_reference_strength": 0.9,
    }
    values.update(overrides)
    return values


class CharacterPlannerHelperTests(unittest.TestCase):
    def test_default_plan_compilation(self):
        result = build_character_plan(make_ui_values())

        self.assertIn("Identity:", result["prompt"])
        self.assertIn("Face Handling:", result["prompt"])
        self.assertEqual(result["system_instructions"], "")
        self.assertIn('"avatar_type": "adult_female_photorealistic"', result["character_plan_json"])
        self.assertIn("Outfit variant: classic_triangle", result["summary"])

    def test_deep_merge_nested_override(self):
        result = build_character_plan(
            make_ui_values(),
            plan_overrides_json=json.dumps(
                {
                    "body": {
                        "frame": {
                            "pelvis_width": 0.61,
                        }
                    }
                }
            ),
        )

        self.assertEqual(result["plan"]["body"]["frame"]["pelvis_width"], 0.61)
        self.assertEqual(result["plan"]["body"]["frame"]["shoulder_breadth"], 0.52)

    def test_json_wrapper_is_stripped(self):
        wrapped = "<JSON>{\"rendering\": {\"lighting\": {\"shadow_strength\": 0.18}}}</JSON>"
        result = build_character_plan(make_ui_values(), plan_overrides_json=wrapped)

        self.assertEqual(result["plan"]["rendering"]["lighting"]["shadow_strength"], 0.18)
        self.assertIn("Overrides applied: yes", result["summary"])

    def test_invalid_json_is_rejected(self):
        with self.assertRaises(ValueError):
            build_character_plan(make_ui_values(), plan_overrides_json="{not valid json}")

    def test_unsupported_key_warning_reporting(self):
        result = build_character_plan(
            make_ui_values(),
            plan_overrides_json=json.dumps(
                {
                    "custom_section": {"foo": 1},
                    "body": {"custom_ratio": 0.5},
                }
            ),
        )

        self.assertIn("custom_section", result["unknown_paths"])
        self.assertIn("body.custom_ratio", result["unknown_paths"])
        self.assertIn("Unsupported override keys not mapped by the prompt compiler", result["summary"])

    def test_face_first_reference_ordering(self):
        manifest = resolve_reference_manifest(
            use_face_reference=True,
            face_reference_present=True,
            extra_reference_batch_sizes=[2, 1],
        )

        self.assertEqual(manifest["reference_labels"][0], "face_reference")
        self.assertEqual(manifest["count"], 4)

    def test_reference_manifest_truncates_to_fourteen(self):
        manifest = resolve_reference_manifest(
            use_face_reference=True,
            face_reference_present=True,
            extra_reference_batch_sizes=[10, 10],
        )

        self.assertEqual(manifest["count"], 14)
        self.assertTrue(manifest["truncated"])
        self.assertIn("Reference images truncated from 21 to 14", manifest["warnings"][0])

    def test_outfit_variant_change_does_not_mutate_body_metrics(self):
        base_result = build_character_plan(make_ui_values(outfit_variant="classic_triangle"))
        variant_result = build_character_plan(make_ui_values(outfit_variant="halter_contour"))

        self.assertEqual(base_result["plan"]["body"]["measurements"], variant_result["plan"]["body"]["measurements"])
        self.assertNotEqual(base_result["plan"]["base_outfit"]["variant"], variant_result["plan"]["base_outfit"]["variant"])

    def test_face_lock_activates_with_connected_face(self):
        result = build_character_plan(
            make_ui_values(use_face_reference=True),
            face_reference_present=True,
            extra_reference_batch_sizes=[1],
        )

        self.assertTrue(result["face_lock_active"])
        self.assertIn("exact facial features and identity", result["system_instructions"])
        self.assertIn("Face lock: active", result["summary"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
