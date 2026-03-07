import json
import unittest
import os
import sys
import importlib
import types as pytypes
from unittest import mock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def install_test_stubs():
    try:
        import torch as _torch  # noqa: F401
    except ModuleNotFoundError:
        torch_stub = pytypes.ModuleType("torch")

        class Tensor:
            def __init__(self, shape):
                self.shape = tuple(shape)
                self.ndim = len(self.shape)

            def clone(self):
                return Tensor(self.shape)

        def zeros(shape):
            return Tensor(shape)

        def ones(shape):
            return Tensor(shape)

        def is_tensor(value):
            return isinstance(value, Tensor)

        def cat(tensors, dim=0):
            if not tensors:
                raise ValueError("cat expects at least one tensor")
            first_shape = list(tensors[0].shape)
            first_shape[dim] = sum(t.shape[dim] for t in tensors)
            return Tensor(first_shape)

        def from_numpy(array):
            return Tensor(getattr(array, "shape", ()))

        torch_stub.Tensor = Tensor
        torch_stub.zeros = zeros
        torch_stub.ones = ones
        torch_stub.cat = cat
        torch_stub.from_numpy = from_numpy
        torch_stub.is_tensor = is_tensor
        sys.modules["torch"] = torch_stub

    try:
        import numpy as _numpy  # noqa: F401
    except ModuleNotFoundError:
        numpy_stub = pytypes.ModuleType("numpy")
        sys.modules["numpy"] = numpy_stub

    try:
        from PIL import Image as _Image  # noqa: F401
    except ModuleNotFoundError:
        pil_stub = pytypes.ModuleType("PIL")

        class ImageStub:
            @staticmethod
            def fromarray(value):
                return value

            @staticmethod
            def open(value):
                return value

        pil_stub.Image = ImageStub
        sys.modules["PIL"] = pil_stub

    try:
        from google import genai as _genai  # noqa: F401
    except ModuleNotFoundError:
        google_stub = sys.modules.get("google")
        if google_stub is None:
            google_stub = pytypes.ModuleType("google")
            sys.modules["google"] = google_stub

        genai_stub = pytypes.ModuleType("google.genai")

        class DummyConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class DummyClient:
            def __init__(self, api_key=None):
                self.api_key = api_key
                self.models = self

            def generate_content(self, *args, **kwargs):
                raise NotImplementedError("Dummy client should be mocked in tests.")

        genai_stub.Client = DummyClient
        genai_stub.types = pytypes.SimpleNamespace(
            ImageConfig=DummyConfig,
            GenerateContentConfig=DummyConfig,
            ThinkingConfig=DummyConfig,
            Tool=DummyConfig,
            GoogleSearch=DummyConfig,
            SearchTypes=DummyConfig,
            WebSearch=DummyConfig,
            ImageSearch=DummyConfig,
        )
        google_stub.genai = genai_stub
        sys.modules["google.genai"] = genai_stub


install_test_stubs()

import torch

with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("network disabled in tests")):
    nodes_module = importlib.import_module("nodes")

BurveCharacterPlanner = nodes_module.BurveCharacterPlanner
BurveCharacterRaceDetails = nodes_module.BurveCharacterRaceDetails
BurveGoogleImageGen = nodes_module.BurveGoogleImageGen
CHARACTER_GEN_PIPE_KIND = nodes_module.CHARACTER_GEN_PIPE_KIND
CHARACTER_GEN_PIPE_VERSION = nodes_module.CHARACTER_GEN_PIPE_VERSION
CHARACTER_RACE_PIPE_KIND = nodes_module.CHARACTER_RACE_PIPE_KIND
CHARACTER_RACE_PIPE_VERSION = nodes_module.CHARACTER_RACE_PIPE_VERSION

from character_planner import build_character_plan, resolve_reference_manifest


def make_ui_values(**overrides):
    values = {
        "gender": "female",
        "age_years": 25,
        "race": "human",
        "custom_race": "",
        "height_cm": 170,
        "weight_kg": 58,
        "bust_cm": 90,
        "underbust_cm": 74,
        "waist_cm": 64,
        "full_hip_cm": 95,
        "male_chest_cm": 102,
        "body_frame_preset": "balanced",
        "male_body_frame_preset": "balanced",
        "skin_tone": "light_medium",
        "custom_skin_tone": "",
        "undertone": "neutral",
        "hair_color": "dark_blonde",
        "custom_hair_color": "",
        "hair_length": "long",
        "musculature_tone": 0.35,
        "body_fat": 0.28,
        "pose": "neutral_a_pose",
        "outfit_variant": "classic_triangle",
        "male_outfit_variant": "classic_brief",
        "outfit_color": "neutral_gray",
        "use_face_reference": False,
        "face_reference_strength": 0.9,
    }
    values.update(overrides)
    return values


def make_race_pipe(**overrides):
    traits = {
        "ears": "",
        "horns": "",
        "wings": "",
        "tail": "",
        "legs_feet": "",
        "skin_surface": "",
        "head_features": "",
        "hands_arms": "",
        "extra_notes": "",
    }
    traits.update(overrides.pop("traits", {}))
    payload = {
        "kind": CHARACTER_RACE_PIPE_KIND,
        "version": CHARACTER_RACE_PIPE_VERSION,
        "race_name": "",
        "traits": traits,
        "summary": "Race name: none",
    }
    payload.update(overrides)
    return payload


class CharacterPlannerHelperTests(unittest.TestCase):
    def test_default_female_plan_compilation(self):
        result = build_character_plan(make_ui_values())

        self.assertIn("Identity:", result["prompt"])
        self.assertIn('"gender": "female"', result["prompt"])
        self.assertEqual(result["system_instructions"], "")
        self.assertEqual(result["plan"]["identity"]["gender"], "female")
        self.assertEqual(result["plan"]["identity"]["age_years"], 25)
        self.assertEqual(result["plan"]["identity"]["race"]["base"], "human")
        self.assertIn("Gender: female", result["summary"])
        self.assertIn("Age: 25 (adult)", result["summary"])
        self.assertIn("Race: human", result["summary"])
        self.assertIn("Fantasy traits: none", result["summary"])
        self.assertIn("Custom text overrides: none", result["summary"])

    def test_age_group_derivation(self):
        result = build_character_plan(make_ui_values(age_years=42))

        self.assertEqual(result["plan"]["identity"]["age_group"], "mature_adult")
        self.assertIn("Age: 42 (mature_adult)", result["summary"])

    def test_underage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity.age_years must be an integer between 18 and 80."):
            build_character_plan(make_ui_values(age_years=17))

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

    def test_custom_hair_color_overrides_dropdown_value(self):
        result = build_character_plan(make_ui_values(custom_hair_color="pink"))

        self.assertEqual(result["plan"]["hair"]["scalp_hair"]["color"], "pink")
        self.assertEqual(result["custom_text_overrides_applied"], ["hair_color"])
        self.assertIn("Custom text overrides: hair_color", result["summary"])

    def test_custom_skin_tone_overrides_dropdown_value(self):
        result = build_character_plan(make_ui_values(custom_skin_tone="green"))

        self.assertEqual(result["plan"]["identity"]["skin_tone"]["base"], "green")
        self.assertEqual(result["custom_text_overrides_applied"], ["skin_tone"])
        self.assertIn("Custom text overrides: skin_tone", result["summary"])

    def test_custom_race_overrides_dropdown_value(self):
        result = build_character_plan(make_ui_values(race="elf", custom_race="moon_elf"))

        self.assertEqual(result["plan"]["identity"]["race"]["base"], "moon_elf")
        self.assertEqual(result["custom_text_overrides_applied"], ["race"])
        self.assertIn("Custom text overrides: race", result["summary"])

    def test_blank_custom_text_overrides_are_ignored(self):
        result = build_character_plan(
            make_ui_values(custom_hair_color="   ", custom_skin_tone="\n\t", custom_race=" ")
        )

        self.assertEqual(result["plan"]["hair"]["scalp_hair"]["color"], "dark_blonde")
        self.assertEqual(result["plan"]["identity"]["skin_tone"]["base"], "light_medium")
        self.assertEqual(result["plan"]["identity"]["race"]["base"], "human")
        self.assertEqual(result["custom_text_overrides_applied"], [])
        self.assertIn("Custom text overrides: none", result["summary"])

    def test_plan_overrides_json_still_wins_over_custom_text_and_race_pipe(self):
        result = build_character_plan(
            make_ui_values(custom_hair_color="pink", custom_skin_tone="green", custom_race="fae"),
            plan_overrides_json=json.dumps(
                {
                    "hair": {
                        "scalp_hair": {
                            "color": "silver",
                        }
                    },
                    "identity": {
                        "skin_tone": {
                            "base": "blue",
                        },
                        "race": {
                            "base": "vampire",
                        },
                    },
                    "fantasy_traits": {
                        "wings": "spectral",
                    },
                }
            ),
            race_override_pipe=make_race_pipe(race_name="dragonkin", traits={"wings": "dragon_membrane"}),
        )

        self.assertEqual(result["plan"]["hair"]["scalp_hair"]["color"], "silver")
        self.assertEqual(result["plan"]["identity"]["skin_tone"]["base"], "blue")
        self.assertEqual(result["plan"]["identity"]["race"]["base"], "vampire")
        self.assertEqual(result["plan"]["fantasy_traits"]["wings"], "spectral")
        self.assertEqual(result["custom_text_overrides_applied"], ["hair_color", "skin_tone", "race"])

    def test_gender_female_keeps_female_chest_and_outfit(self):
        result = build_character_plan(make_ui_values(gender="female"))

        self.assertEqual(result["plan"]["avatar_type"], "adult_female_photorealistic")
        self.assertIn("chest", result["plan"])
        self.assertNotIn("male_torso", result["plan"])
        self.assertNotEqual(result["plan"]["base_outfit"]["top"]["type"], "none")
        self.assertEqual(
            result["ignored_gender_specific_controls"],
            ["male_chest_cm", "male_body_frame_preset", "male_outfit_variant"],
        )

    def test_gender_male_uses_male_torso_and_no_top(self):
        result = build_character_plan(
            make_ui_values(
                gender="male",
                male_body_frame_preset="v_taper",
                male_outfit_variant="square_cut",
                male_chest_cm=108,
            )
        )

        self.assertEqual(result["plan"]["avatar_type"], "adult_male_photorealistic")
        self.assertIn("male_torso", result["plan"])
        self.assertNotIn("chest", result["plan"])
        self.assertEqual(result["plan"]["male_torso"]["measurements"]["chest_cm"], 108)
        self.assertEqual(result["plan"]["base_outfit"]["variant"], "square_cut")
        self.assertEqual(result["plan"]["base_outfit"]["top"]["type"], "none")
        self.assertIn("Gender: male", result["summary"])
        self.assertIn(
            "Ignored gender-specific controls: bust_cm, underbust_cm, body_frame_preset, outfit_variant",
            result["summary"],
        )

    def test_male_plan_with_bra_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "male plans must keep base_outfit.top.type set to 'none'."):
            build_character_plan(
                make_ui_values(gender="male"),
                plan_overrides_json=json.dumps(
                    {
                        "base_outfit": {
                            "top": {
                                "type": "triangle_bikini",
                            }
                        }
                    }
                ),
            )

    def test_race_pipe_traits_are_merged(self):
        result = build_character_plan(
            make_ui_values(),
            race_override_pipe=make_race_pipe(
                race_name="tiefling",
                traits={
                    "horns": "ram",
                    "wings": "bat_leather",
                },
            ),
        )

        self.assertEqual(result["plan"]["identity"]["race"]["base"], "tiefling")
        self.assertEqual(result["plan"]["fantasy_traits"]["horns"], "ram")
        self.assertEqual(result["plan"]["fantasy_traits"]["wings"], "bat_leather")
        self.assertIn("Race: tiefling", result["summary"])
        self.assertIn("Fantasy traits: horns=ram, wings=bat_leather", result["summary"])

    def test_malformed_race_pipe_is_rejected(self):
        malformed = make_race_pipe(kind="wrong.kind")

        with self.assertRaisesRegex(ValueError, "Invalid race_override_pipe: unexpected kind."):
            build_character_plan(make_ui_values(), race_override_pipe=malformed)


class CharacterPlannerNodeTests(unittest.TestCase):
    def test_planner_emits_character_pipe(self):
        planner = BurveCharacterPlanner()

        outputs = planner.build(**make_ui_values())
        prompt, system_instructions, reference_images, character_plan_json, summary, character_pipe = outputs

        self.assertEqual(len(outputs), 6)
        self.assertEqual(character_pipe["kind"], CHARACTER_GEN_PIPE_KIND)
        self.assertEqual(character_pipe["version"], CHARACTER_GEN_PIPE_VERSION)
        self.assertEqual(character_pipe["prompt"], prompt)
        self.assertEqual(character_pipe["system_instructions"], system_instructions)
        self.assertIs(character_pipe["reference_images"], reference_images)
        self.assertEqual(character_pipe["character_plan_json"], character_plan_json)
        self.assertEqual(character_pipe["summary"], summary)

    def test_planner_preserves_first_five_outputs(self):
        planner = BurveCharacterPlanner()
        expected = build_character_plan(make_ui_values())

        outputs = planner.build(**make_ui_values())

        self.assertEqual(outputs[0], expected["prompt"])
        self.assertEqual(outputs[1], expected["system_instructions"])
        self.assertEqual(outputs[2], [])
        self.assertEqual(outputs[3], expected["character_plan_json"])
        self.assertEqual(outputs[4], expected["summary"])

    def test_input_types_place_custom_overrides_next_to_controls(self):
        required_keys = list(BurveCharacterPlanner.INPUT_TYPES()["required"].keys())

        self.assertEqual(required_keys[required_keys.index("race") + 1], "custom_race")
        self.assertEqual(required_keys[required_keys.index("skin_tone") + 1], "custom_skin_tone")
        self.assertEqual(required_keys[required_keys.index("hair_color") + 1], "custom_hair_color")


class CharacterRaceDetailsNodeTests(unittest.TestCase):
    def test_race_details_node_emits_valid_pipe(self):
        node = BurveCharacterRaceDetails()

        race_override_pipe, race_override_json, summary = node.build(
            race_name="demon",
            custom_race_name="",
            ears="none",
            custom_ears="",
            horns="ram",
            custom_horns="",
            wings="bat_leather",
            custom_wings="",
            tail="none",
            custom_tail="",
            legs_feet="none",
            custom_legs_feet="",
            skin_surface="none",
            custom_skin_surface="",
            head_features="fangs",
            custom_head_features="",
            hands_arms="claws",
            custom_hands_arms="",
            extra_notes="",
        )

        self.assertEqual(race_override_pipe["kind"], CHARACTER_RACE_PIPE_KIND)
        self.assertEqual(race_override_pipe["version"], CHARACTER_RACE_PIPE_VERSION)
        self.assertEqual(race_override_pipe["race_name"], "demon")
        self.assertEqual(race_override_pipe["traits"]["horns"], "ram")
        self.assertEqual(race_override_pipe["traits"]["wings"], "bat_leather")
        self.assertEqual(json.loads(race_override_json)["race_name"], "demon")
        self.assertIn("Active traits: horns=ram, wings=bat_leather", summary)

    def test_race_details_custom_fields_override_dropdowns(self):
        node = BurveCharacterRaceDetails()

        race_override_pipe, _, summary = node.build(
            race_name="angel",
            custom_race_name="custom_seraph",
            ears="pointed_elf",
            custom_ears="custom ears",
            horns="none",
            custom_horns="crown horns of light",
            wings="large_feathered",
            custom_wings="",
            tail="none",
            custom_tail="",
            legs_feet="none",
            custom_legs_feet="",
            skin_surface="none",
            custom_skin_surface="",
            head_features="none",
            custom_head_features="",
            hands_arms="none",
            custom_hands_arms="",
            extra_notes="floating halo",
        )

        self.assertEqual(race_override_pipe["race_name"], "custom_seraph")
        self.assertEqual(race_override_pipe["traits"]["ears"], "custom ears")
        self.assertEqual(race_override_pipe["traits"]["horns"], "crown horns of light")
        self.assertEqual(race_override_pipe["traits"]["wings"], "large_feathered")
        self.assertEqual(race_override_pipe["traits"]["extra_notes"], "floating halo")
        self.assertIn("Custom text overrides: race_name, ears, horns, extra_notes", summary)

    def test_race_details_none_and_blank_custom_leaves_traits_unset(self):
        node = BurveCharacterRaceDetails()

        race_override_pipe, _, summary = node.build(
            race_name="none",
            custom_race_name="",
            ears="none",
            custom_ears="",
            horns="none",
            custom_horns="",
            wings="none",
            custom_wings="",
            tail="none",
            custom_tail="",
            legs_feet="none",
            custom_legs_feet="",
            skin_surface="none",
            custom_skin_surface="",
            head_features="none",
            custom_head_features="",
            hands_arms="none",
            custom_hands_arms="",
            extra_notes="",
        )

        self.assertEqual(race_override_pipe["race_name"], "")
        self.assertTrue(all(value == "" for value in race_override_pipe["traits"].values()))
        self.assertIn("Active traits: none", summary)


class BurveGoogleImageGenResolverTests(unittest.TestCase):
    def setUp(self):
        self.node = BurveGoogleImageGen()
        self.pipe_reference = torch.ones((1, 2, 2, 3))
        self.direct_reference = torch.zeros((1, 2, 2, 3))
        self.character_pipe = {
            "kind": CHARACTER_GEN_PIPE_KIND,
            "version": CHARACTER_GEN_PIPE_VERSION,
            "prompt": "pipe prompt",
            "system_instructions": "pipe system",
            "reference_images": [self.pipe_reference],
            "character_plan_json": "{\"avatar_type\": \"adult_female_photorealistic\"}",
            "summary": "Gender: female\nFace lock: inactive",
        }

    def test_resolver_uses_pipe_values_when_direct_inputs_are_blank(self):
        resolved = self.node._resolve_generation_inputs(
            prompt="",
            system_instructions="",
            reference_images=None,
            character_pipe=self.character_pipe,
        )

        self.assertEqual(resolved["prompt"], "pipe prompt")
        self.assertEqual(resolved["system_instructions"], "pipe system")
        self.assertIs(resolved["reference_images"], self.character_pipe["reference_images"])
        self.assertEqual(resolved["character_plan_json"], self.character_pipe["character_plan_json"])
        self.assertEqual(resolved["planner_summary"], self.character_pipe["summary"])
        self.assertFalse(resolved["ignored_direct_inputs"])

    def test_resolver_pipe_overrides_direct_prompt(self):
        resolved = self.node._resolve_generation_inputs(
            prompt="direct prompt",
            system_instructions="",
            reference_images=None,
            character_pipe=self.character_pipe,
        )

        self.assertEqual(resolved["prompt"], "pipe prompt")
        self.assertTrue(resolved["ignored_direct_inputs"])

    def test_resolver_pipe_overrides_direct_system_instructions(self):
        resolved = self.node._resolve_generation_inputs(
            prompt="direct prompt",
            system_instructions="direct system",
            reference_images=None,
            character_pipe=self.character_pipe,
        )

        self.assertEqual(resolved["system_instructions"], "pipe system")
        self.assertTrue(resolved["ignored_direct_inputs"])

    def test_resolver_pipe_overrides_direct_reference_images(self):
        direct_reference_images = [self.direct_reference]

        resolved = self.node._resolve_generation_inputs(
            prompt="direct prompt",
            system_instructions="",
            reference_images=direct_reference_images,
            character_pipe=self.character_pipe,
        )

        self.assertIs(resolved["reference_images"], self.character_pipe["reference_images"])
        self.assertTrue(resolved["ignored_direct_inputs"])

    def test_resolver_without_pipe_preserves_direct_input_behavior(self):
        direct_reference_images = [self.direct_reference]

        resolved = self.node._resolve_generation_inputs(
            prompt="direct prompt",
            system_instructions="direct system",
            reference_images=direct_reference_images,
            character_pipe=None,
        )

        self.assertEqual(resolved["prompt"], "direct prompt")
        self.assertEqual(resolved["system_instructions"], "direct system")
        self.assertIs(resolved["reference_images"], direct_reference_images)
        self.assertFalse(resolved["ignored_direct_inputs"])

    def test_resolver_rejects_empty_prompt_without_pipe_prompt(self):
        empty_prompt_pipe = dict(self.character_pipe, prompt="   ")

        with self.assertRaisesRegex(ValueError, "Prompt is empty. Provide a prompt or connect character_pipe."):
            self.node._resolve_generation_inputs(
                prompt="",
                system_instructions="",
                reference_images=None,
                character_pipe=empty_prompt_pipe,
            )

    def test_resolver_rejects_unexpected_pipe_kind(self):
        malformed_pipe = dict(self.character_pipe, kind="wrong.kind")

        with self.assertRaisesRegex(ValueError, "Invalid character_pipe: unexpected kind."):
            self.node._resolve_generation_inputs(
                prompt="",
                system_instructions="",
                reference_images=None,
                character_pipe=malformed_pipe,
            )

    def test_resolver_rejects_unsupported_pipe_version(self):
        malformed_pipe = dict(self.character_pipe, version=999)

        with self.assertRaisesRegex(ValueError, "Invalid character_pipe: unsupported version."):
            self.node._resolve_generation_inputs(
                prompt="",
                system_instructions="",
                reference_images=None,
                character_pipe=malformed_pipe,
            )

    def test_generate_image_appends_override_note_and_planner_summary(self):
        class FakeResponse:
            candidates = None
            parts = []

        class FakeClient:
            last_call = None

            def __init__(self, api_key):
                self.api_key = api_key
                self.models = self

            def generate_content(self, model, contents, config):
                FakeClient.last_call = {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
                return FakeResponse()

        pipe_without_refs = dict(self.character_pipe, reference_images=None)
        with mock.patch.object(nodes_module.genai, "Client", FakeClient):
            with mock.patch.object(self.node, "_get_api_key_or_error", return_value=("test-key", None)):
                image, thinking_image, thinking_process, system_messages = self.node.generate_image(
                    prompt="direct prompt",
                    model="gemini-2.5-flash-image",
                    resolution="1K",
                    aspect_ratio="1:1",
                    seed=123,
                    search_mode="off",
                    enable_google_search=False,
                    enable_image_search=False,
                    thinking_mode="legacy_toggle",
                    enable_thinking_mode=True,
                    system_instructions="direct system",
                    reference_images=None,
                    character_pipe=pipe_without_refs,
                )

        self.assertEqual(FakeClient.last_call["contents"][0], "pipe prompt")
        self.assertIn(
            "character_pipe is connected; ignoring direct prompt, system_instructions, and reference_images inputs.",
            system_messages,
        )
        self.assertIn("Planner summary:\nGender: female", system_messages)
        self.assertTrue(torch.is_tensor(image))
        self.assertTrue(torch.is_tensor(thinking_image))
        self.assertEqual(thinking_process, "")

    def test_generate_image_skips_override_note_when_no_direct_inputs_were_ignored(self):
        class FakeResponse:
            candidates = None
            parts = []

        class FakeClient:
            last_call = None

            def __init__(self, api_key):
                self.api_key = api_key
                self.models = self

            def generate_content(self, model, contents, config):
                FakeClient.last_call = {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
                return FakeResponse()

        pipe_without_refs = dict(self.character_pipe, reference_images=None)
        with mock.patch.object(nodes_module.genai, "Client", FakeClient):
            with mock.patch.object(self.node, "_get_api_key_or_error", return_value=("test-key", None)):
                _, _, _, system_messages = self.node.generate_image(
                    prompt="",
                    model="gemini-2.5-flash-image",
                    resolution="1K",
                    aspect_ratio="1:1",
                    seed=123,
                    search_mode="off",
                    enable_google_search=False,
                    enable_image_search=False,
                    thinking_mode="legacy_toggle",
                    enable_thinking_mode=True,
                    system_instructions="",
                    reference_images=None,
                    character_pipe=pipe_without_refs,
                )

        self.assertNotIn("ignoring direct prompt", system_messages)
        self.assertIn("Planner summary:\nGender: female", system_messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
