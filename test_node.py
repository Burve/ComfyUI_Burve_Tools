import asyncio
import json
import unittest
import os
import sys
import importlib
import tempfile
import types as pytypes
from unittest import mock

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(REPO_DIR)
PACKAGE_NAME = os.path.basename(REPO_DIR)

sys.path.append(REPO_DIR)
sys.path.append(PARENT_DIR)


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

    google_stub = sys.modules.get("google")
    if google_stub is None:
        google_stub = pytypes.ModuleType("google")
        sys.modules["google"] = google_stub

    genai_stub = pytypes.ModuleType("google.genai")

    class DummyConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyClient:
        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key
            self.kwargs = kwargs
            self.models = self

        def generate_content(self, *args, **kwargs):
            raise NotImplementedError("Dummy client should be mocked in tests.")

    class DummyThinkingLevel:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        MINIMAL = "MINIMAL"

    class DummyProminentPeople:
        ALLOW_PROMINENT_PEOPLE = "ALLOW_PROMINENT_PEOPLE"
        BLOCK_PROMINENT_PEOPLE = "BLOCK_PROMINENT_PEOPLE"

    class DummyAPIError(Exception):
        def __init__(self, code, response_json, response=None):
            self.code = code
            self.details = response_json
            self.response = response
            self.status = ""
            super().__init__(str(response_json))

    genai_stub.__version__ = "1.68.0"
    genai_stub.Client = DummyClient
    genai_stub.types = pytypes.SimpleNamespace(
        ImageConfig=DummyConfig,
        GenerateContentConfig=DummyConfig,
        HttpOptions=DummyConfig,
        ThinkingConfig=DummyConfig,
        Tool=DummyConfig,
        GoogleSearch=DummyConfig,
        SearchTypes=DummyConfig,
        WebSearch=DummyConfig,
        ImageSearch=DummyConfig,
        ThinkingLevel=DummyThinkingLevel,
        ProminentPeople=DummyProminentPeople,
    )
    genai_errors_stub = pytypes.ModuleType("google.genai.errors")
    genai_errors_stub.APIError = DummyAPIError
    genai_errors_stub.ClientError = DummyAPIError
    genai_errors_stub.ServerError = DummyAPIError
    genai_stub.errors = genai_errors_stub
    google_stub.genai = genai_stub
    sys.modules["google.genai"] = genai_stub
    sys.modules["google.genai.errors"] = genai_errors_stub

    if "comfy_api.latest" not in sys.modules:
        comfy_api_stub = pytypes.ModuleType("comfy_api")
        latest_stub = pytypes.ModuleType("comfy_api.latest")

        class ControlAfterGenerate:
            fixed = "fixed"
            increment = "increment"
            decrement = "decrement"
            randomize = "randomize"

        class Hidden:
            unique_id = "UNIQUE_ID"
            prompt = "PROMPT"
            extra_pnginfo = "EXTRA_PNGINFO"
            dynprompt = "DYNPROMPT"
            auth_token_comfy_org = "AUTH_TOKEN_COMFY_ORG"
            api_key_comfy_org = "API_KEY_COMFY_ORG"

        class StubInput:
            def __init__(self, id, **kwargs):
                self.id = id
                self.display_name = kwargs.get("display_name")
                self.optional = kwargs.get("optional", False)
                self.tooltip = kwargs.get("tooltip")
                self.extra_dict = kwargs.get("extra_dict")
                self.kwargs = kwargs

        class StubOutput:
            def __init__(self, id=None, display_name=None, tooltip=None, is_output_list=False, **kwargs):
                self.id = id
                self.display_name = display_name
                self.tooltip = tooltip
                self.is_output_list = is_output_list
                self.kwargs = kwargs

        def make_type(io_type):
            class _Type:
                class Input(StubInput):
                    def __init__(self, id, **kwargs):
                        super().__init__(id, **kwargs)
                        self.io_type = io_type

                class Output(StubOutput):
                    def __init__(self, id=None, display_name=None, tooltip=None, is_output_list=False, **kwargs):
                        super().__init__(id=id, display_name=display_name, tooltip=tooltip, is_output_list=is_output_list, **kwargs)
                        self.io_type = io_type

            _Type.io_type = io_type
            return _Type

        class ComboType(make_type("COMBO")):
            class Input(StubInput):
                def __init__(self, id, options=None, **kwargs):
                    super().__init__(id, **kwargs)
                    self.io_type = "COMBO"
                    self.options = list(options or [])

        class DynamicComboType:
            class Option:
                def __init__(self, key, inputs):
                    self.key = key
                    self.inputs = inputs

            class Input(StubInput):
                def __init__(self, id, options=None, **kwargs):
                    super().__init__(id, **kwargs)
                    self.io_type = "COMFY_DYNAMICCOMBO_V3"
                    self.options = list(options or [])

        def Custom(io_type):
            return make_type(io_type)

        class Schema:
            def __init__(self, node_id, display_name=None, category="sd", inputs=None, outputs=None, hidden=None, is_output_node=False, is_input_list=False, not_idempotent=False, **kwargs):
                self.node_id = node_id
                self.display_name = display_name
                self.category = category
                self.inputs = list(inputs or [])
                self.outputs = list(outputs or [])
                self.hidden = list(hidden or [])
                self.is_output_node = is_output_node
                self.is_input_list = is_input_list
                self.not_idempotent = not_idempotent
                self.kwargs = kwargs

            def get_v1_info(self, include_hidden=False):
                required = {}
                optional = {}

                for item in self.inputs:
                    meta = {key: value for key, value in item.kwargs.items() if key != "optional"}
                    if getattr(item, "io_type", None) == "COMBO":
                        io_definition = list(getattr(item, "options", []))
                    else:
                        io_definition = getattr(item, "io_type", None)
                        if hasattr(item, "options"):
                            meta["options"] = list(item.options)

                    target = optional if getattr(item, "optional", False) else required
                    target[item.id] = [io_definition, meta]

                return {
                    "input": {
                        "required": required,
                        "optional": optional,
                    },
                    "input_order": {
                        "required": list(required.keys()),
                        "optional": list(optional.keys()),
                    },
                }

        class NodeOutput:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.result = args if args else None
                self.ui = kwargs.get("ui")
                self.expand = kwargs.get("expand")
                self.block_execution = kwargs.get("block_execution")

        class ComfyNode:
            pass

        class ComfyExtension:
            async def on_load(self):
                return None

            async def get_node_list(self):
                raise NotImplementedError

        io_namespace = pytypes.SimpleNamespace(
            Boolean=make_type("BOOLEAN"),
            Combo=ComboType,
            ComfyNode=ComfyNode,
            ControlAfterGenerate=ControlAfterGenerate,
            Custom=Custom,
            DynamicCombo=DynamicComboType,
            Float=make_type("FLOAT"),
            Hidden=Hidden,
            Image=make_type("IMAGE"),
            Int=make_type("INT"),
            Mask=make_type("MASK"),
            NodeOutput=NodeOutput,
            Schema=Schema,
            String=make_type("STRING"),
        )

        latest_stub.ComfyExtension = ComfyExtension
        latest_stub.IO = io_namespace
        latest_stub.io = io_namespace
        comfy_api_stub.latest = latest_stub
        sys.modules["comfy_api"] = comfy_api_stub
        sys.modules["comfy_api.latest"] = latest_stub


install_test_stubs()

import torch

with mock.patch("urllib.request.urlopen", side_effect=RuntimeError("network disabled in tests")):
    nodes_module = importlib.import_module("nodes")
    gemini_service_module = importlib.import_module("gemini_image_service")
    package_module = importlib.import_module(PACKAGE_NAME)

BurveCharacterPlanner = nodes_module.BurveCharacterPlanner
BurveCharacterRaceDetails = nodes_module.BurveCharacterRaceDetails
BurveGoogleImageGen = nodes_module.BurveGoogleImageGen
BurveVertexImageGen = nodes_module.BurveVertexImageGen
BurveDebugGeminiKey = nodes_module.BurveDebugGeminiKey
BurveDebugVertexAuth = nodes_module.BurveDebugVertexAuth
BurveImageInfo = nodes_module.BurveImageInfo
BurveSaveGeneratedImage = nodes_module.BurveSaveGeneratedImage
CHARACTER_GEN_PIPE_KIND = nodes_module.CHARACTER_GEN_PIPE_KIND
CHARACTER_GEN_PIPE_VERSION = nodes_module.CHARACTER_GEN_PIPE_VERSION
GENERATED_IMAGE_PIPE_KIND = nodes_module.GENERATED_IMAGE_PIPE_KIND
GENERATED_IMAGE_PIPE_VERSION = nodes_module.GENERATED_IMAGE_PIPE_VERSION
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


class SharedImageGenResolverTestMixin:
    NodeClass = None

    def setUp(self):
        self.node = self.NodeClass()
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

    class _FakeClient:
        def __init__(self, response):
            self.response = response
            self.models = self
            self.last_call = None

        def generate_content(self, model, contents, config):
            self.last_call = {
                "model": model,
                "contents": contents,
                "config": config,
            }
            return self.response

    @staticmethod
    def _make_candidate(parts, finish_reason=None, safety_ratings=None):
        return pytypes.SimpleNamespace(
            content=pytypes.SimpleNamespace(parts=parts),
            finish_reason=finish_reason,
            safety_ratings=safety_ratings,
        )

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
        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=FakeClient("test-key")):
                image, thinking_image, thinking_process, system_messages, generated_image_pipe = self.node.generate_image(
                    prompt="direct prompt",
                    model="gemini-2.5-flash-image",
                    seed=123,
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
        self.assertEqual(
            thinking_process,
            gemini_service_module.THINKING_PROCESS_DEFAULT,
        )
        self.assertEqual(generated_image_pipe["kind"], nodes_module.GENERATED_IMAGE_PIPE_KIND)

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
        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=FakeClient("test-key")):
                _, _, _, system_messages, _ = self.node.generate_image(
                    prompt="",
                    model="gemini-2.5-flash-image",
                    seed=123,
                    system_instructions="",
                    reference_images=None,
                    character_pipe=pipe_without_refs,
                )

        self.assertNotIn("ignoring direct prompt", system_messages)
        self.assertIn("Planner summary:\nGender: female", system_messages)

    def test_gemini_25_requests_text_and_image_modalities(self):
        response = pytypes.SimpleNamespace(candidates=None, parts=[])
        fake_client = self._FakeClient(response)

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                self.node.generate_image(
                    prompt="prompt",
                    model="gemini-2.5-flash-image",
                    seed=123,
                    system_instructions="",
                    reference_images=None,
                    character_pipe=None,
                )

        self.assertEqual(fake_client.last_call["config"].response_modalities, ["TEXT", "IMAGE"])

    def test_generate_image_surfaces_text_only_no_image_diagnostics(self):
        response = pytypes.SimpleNamespace(
            candidates=[
                self._make_candidate(
                    parts=[pytypes.SimpleNamespace(text="Model returned text only.", thought=False)],
                    finish_reason="STOP",
                    safety_ratings="safe",
                )
            ],
            prompt_feedback="No blocking feedback.",
        )
        fake_client = self._FakeClient(response)

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                image, thinking_image, thinking_process, system_messages, generated_image_pipe = self.node.generate_image(
                    prompt="prompt",
                    model="gemini-2.5-flash-image",
                    seed=123,
                    system_instructions="",
                    reference_images=None,
                    character_pipe=None,
                )

        self.assertTrue(torch.is_tensor(image))
        self.assertTrue(torch.is_tensor(thinking_image))
        self.assertEqual(
            thinking_process,
            gemini_service_module.THINKING_PROCESS_DEFAULT,
        )
        self.assertIn("Model returned text only.", system_messages)
        self.assertIn("No non-thinking image generated.", system_messages)
        self.assertIn("The request completed, but no usable image part was found in the response.", system_messages)
        self.assertIn("Text response was returned without an image.", system_messages)
        self.assertIn("Finish reason: STOP", system_messages)
        self.assertIn("Candidate feedback: safe", system_messages)
        self.assertIn("Prompt feedback: No blocking feedback.", system_messages)
        self.assertEqual(generated_image_pipe["items"], [])

    def test_generate_image_surfaces_3_pro_thought_text_without_request_thinking_config(self):
        response = pytypes.SimpleNamespace(
            candidates=[
                self._make_candidate(
                    parts=[
                        pytypes.SimpleNamespace(text="scratchpad", thought=True, inline_data=None),
                        pytypes.SimpleNamespace(text="Visible answer.", thought=False, inline_data=None),
                    ],
                    finish_reason="STOP",
                )
            ],
            prompt_feedback=None,
        )
        fake_client = self._FakeClient(response)

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                _, _, thinking_process, system_messages, _ = self.node.generate_image(
                    prompt="prompt",
                    model={
                        "model": "gemini-3-pro-image-preview",
                        "aspect_ratio": "1:1",
                        "resolution": "1K",
                        "search_mode": "off",
                    },
                    seed=123,
                    system_instructions="",
                    reference_images=None,
                    character_pipe=None,
                )

        self.assertEqual(thinking_process, "scratchpad")
        self.assertIsNone(getattr(fake_client.last_call["config"], "thinking_config", None))
        self.assertIn("Visible answer.", system_messages)

    def test_generate_image_promotes_3_pro_thinking_image_when_no_normal_image_exists(self):
        thought_image_bytes = b"thought-image"
        response = pytypes.SimpleNamespace(
            candidates=[
                self._make_candidate(
                    parts=[
                        pytypes.SimpleNamespace(
                            inline_data=pytypes.SimpleNamespace(data=thought_image_bytes),
                            thought=True,
                            text=None,
                        )
                    ],
                    finish_reason="STOP",
                )
            ],
            prompt_feedback=None,
        )
        fake_client = self._FakeClient(response)
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image
        fake_array = mock.Mock()
        fake_array.astype.return_value = fake_array
        fake_array.ndim = 3
        tensor = torch.ones((2, 2, 3))

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
                    with mock.patch.object(gemini_service_module.np, "array", return_value=fake_array, create=True):
                        with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                            image, thinking_image, _, system_messages, generated_image_pipe = self.node.generate_image(
                                prompt="prompt",
                                model={
                                    "model": "gemini-3-pro-image-preview",
                                    "aspect_ratio": "1:1",
                                    "resolution": "1K",
                                    "search_mode": "off",
                                },
                                seed=123,
                                system_instructions="",
                                reference_images=None,
                                character_pipe=None,
                            )

        self.assertTrue(torch.is_tensor(image))
        self.assertTrue(torch.is_tensor(thinking_image))
        self.assertIsNone(getattr(fake_client.last_call["config"], "thinking_config", None))
        self.assertNotIn("No non-thinking image generated.", system_messages)
        self.assertEqual(generated_image_pipe["items"], [])

    def test_generate_image_returns_timeout_error_for_stalled_request(self):
        class FakeClient:
            def __init__(self):
                self.models = self
                self.call_count = 0

            def generate_content(self, model, contents, config):
                self.call_count += 1
                raise TimeoutError("timed out while waiting")

        fake_client = FakeClient()
        monotonic_values = iter([0.0, 0.0, 121.0])

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(nodes_module.time, "monotonic", side_effect=lambda: next(monotonic_values)):
                    with mock.patch.object(nodes_module.time, "sleep", return_value=None):
                        image, thinking_image, _, system_messages, _ = self.node.generate_image(
                            prompt="prompt",
                            model="gemini-2.5-flash-image",
                            seed=123,
                            request_timeout_seconds=120,
                            retry_attempts=5,
                        )

        self.assertEqual(fake_client.call_count, 1)
        self.assertTrue(torch.is_tensor(image))
        self.assertTrue(torch.is_tensor(thinking_image))
        self.assertIn("Error: Request timed out after 120s waiting for Gemini image generation.", system_messages)
        self.assertIn("Attempts completed: 1.", system_messages)

    def test_generate_image_retries_transient_errors_with_shrinking_timeout_budget(self):
        response = pytypes.SimpleNamespace(candidates=None, parts=[])

        class FakeClient:
            def __init__(self):
                self.models = self
                self.call_count = 0
                self.config_timeouts = []

            def generate_content(self, model, contents, config):
                self.call_count += 1
                self.config_timeouts.append(getattr(config.http_options, "timeout", None))
                if self.call_count == 1:
                    raise nodes_module.genai_errors.APIError(429, "busy")
                return response

        fake_client = FakeClient()
        monotonic_values = iter([0.0, 0.0, 1.0, 2.0])

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(nodes_module.time, "monotonic", side_effect=lambda: next(monotonic_values)):
                    with mock.patch.object(nodes_module.time, "sleep", return_value=None):
                        with mock.patch.object(nodes_module.random, "uniform", return_value=1.0):
                            _, _, thinking_process, system_messages, _ = self.node.generate_image(
                                prompt="prompt",
                                model="gemini-2.5-flash-image",
                                seed=123,
                                request_timeout_seconds=120,
                                retry_attempts=5,
                            )

        self.assertEqual(fake_client.call_count, 2)
        self.assertEqual(thinking_process, gemini_service_module.THINKING_PROCESS_DEFAULT)
        self.assertIn("No non-thinking image generated.", system_messages)
        self.assertEqual(fake_client.config_timeouts[0], 120000)
        self.assertEqual(fake_client.config_timeouts[1], 118000)

    def test_generate_image_does_not_retry_non_retryable_api_errors(self):
        class FakeClient:
            def __init__(self):
                self.models = self
                self.call_count = 0

            def generate_content(self, model, contents, config):
                self.call_count += 1
                raise nodes_module.genai_errors.APIError(400, "bad request")

        fake_client = FakeClient()
        monotonic_values = iter([0.0, 0.0])

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(nodes_module.time, "monotonic", side_effect=lambda: next(monotonic_values)):
                    image, thinking_image, _, system_messages, _ = self.node.generate_image(
                        prompt="prompt",
                        model="gemini-2.5-flash-image",
                        seed=123,
                        request_timeout_seconds=120,
                        retry_attempts=5,
                    )

        self.assertEqual(fake_client.call_count, 1)
        self.assertTrue(torch.is_tensor(image))
        self.assertTrue(torch.is_tensor(thinking_image))
        self.assertIn("Error: bad request", system_messages)
        self.assertNotIn("Request failed after", system_messages)

    def test_generate_image_reports_retry_exhaustion_for_transient_errors(self):
        class FakeClient:
            def __init__(self):
                self.models = self
                self.call_count = 0

            def generate_content(self, model, contents, config):
                self.call_count += 1
                raise nodes_module.genai_errors.APIError(429, "busy")

        fake_client = FakeClient()
        monotonic_values = iter([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(nodes_module.time, "monotonic", side_effect=lambda: next(monotonic_values)):
                    with mock.patch.object(nodes_module.time, "sleep", return_value=None):
                        with mock.patch.object(nodes_module.random, "uniform", return_value=1.0):
                            _, _, _, system_messages, _ = self.node.generate_image(
                                prompt="prompt",
                                model="gemini-2.5-flash-image",
                                seed=123,
                                request_timeout_seconds=120,
                                retry_attempts=3,
                            )

        self.assertEqual(fake_client.call_count, 3)
        self.assertIn("Error: Request failed after 3 attempts.", system_messages)
        self.assertIn("Last error: status 429", system_messages)

    def test_timeout_error_preserves_planner_summary_and_override_note(self):
        class FakeClient:
            def __init__(self):
                self.models = self

            def generate_content(self, model, contents, config):
                raise TimeoutError("timed out while waiting")

        fake_client = FakeClient()
        monotonic_values = iter([0.0, 0.0, 121.0])
        pipe_without_refs = dict(self.character_pipe, reference_images=None)

        with mock.patch.object(self.node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(self.node, "_build_client", return_value=fake_client):
                with mock.patch.object(nodes_module.time, "monotonic", side_effect=lambda: next(monotonic_values)):
                    with mock.patch.object(nodes_module.time, "sleep", return_value=None):
                        _, _, _, system_messages, _ = self.node.generate_image(
                            prompt="direct prompt",
                            model="gemini-2.5-flash-image",
                            seed=123,
                            system_instructions="direct system",
                            reference_images=None,
                            character_pipe=pipe_without_refs,
                            request_timeout_seconds=120,
                            retry_attempts=5,
                        )

        self.assertIn(
            "character_pipe is connected; ignoring direct prompt, system_instructions, and reference_images inputs.",
            system_messages,
        )
        self.assertIn("Planner summary:\nGender: female", system_messages)
        self.assertIn("Error: Request timed out after 120s waiting for Gemini image generation.", system_messages)


class BurveGoogleImageGenResolverTests(SharedImageGenResolverTestMixin, unittest.TestCase):
    NodeClass = BurveGoogleImageGen


class BurveVertexImageGenResolverTests(SharedImageGenResolverTestMixin, unittest.TestCase):
    NodeClass = BurveVertexImageGen


class GeminiImageServiceRequestBuilderTests(unittest.TestCase):
    def test_gemini_25_request_builder_emits_no_resolution_search_or_thinking_config(self):
        request = gemini_service_module.prepare_generate_content_request(
            provider_id="aistudio",
            prompt="prompt",
            model={
                "model": "gemini-2.5-flash-image",
                "aspect_ratio": "16:9",
            },
            seed=123,
        )

        self.assertEqual(request.model_id, "gemini-2.5-flash-image")
        self.assertEqual(request.config.response_modalities, ["TEXT", "IMAGE"])
        self.assertEqual(request.config.image_config.aspect_ratio, "16:9")
        self.assertIsNone(getattr(request.config.image_config, "image_size", None))
        self.assertIsNone(getattr(request.config, "tools", None))
        self.assertIsNone(getattr(request.config, "thinking_config", None))

    def test_gemini_3_pro_request_builder_emits_web_search_without_thinking_config(self):
        request = gemini_service_module.prepare_generate_content_request(
            provider_id="vertex",
            prompt="prompt",
            model={
                "model": "gemini-3-pro-image-preview",
                "aspect_ratio": "1:1",
                "resolution": "2K",
                "search_mode": "web",
            },
            seed=123,
        )

        self.assertEqual(request.config.image_config.image_size, "2K")
        self.assertEqual(request.config.tools[0].google_search.__class__.__name__, "DummyConfig")
        self.assertIsNone(getattr(request.config, "thinking_config", None))

    def test_gemini_3_1_request_builder_emits_search_types_and_output_controls(self):
        request = gemini_service_module.prepare_generate_content_request(
            provider_id="vertex",
            prompt="prompt",
            model={
                "model": "gemini-3.1-flash-image-preview",
                "aspect_ratio": "4:1",
                "resolution": "512",
                "search_mode": "web+image",
                "thinking_level": "MINIMAL",
                "include_thoughts": True,
                "output_mime_type": "image/webp",
                "prominent_people": "block",
            },
            seed=123,
        )

        self.assertEqual(request.config.image_config.aspect_ratio, "4:1")
        self.assertEqual(request.config.image_config.image_size, "512")
        self.assertEqual(request.config.image_config.output_mime_type, "image/webp")
        self.assertEqual(
            request.config.image_config.prominent_people,
            "BLOCK_PROMINENT_PEOPLE",
        )
        self.assertEqual(request.config.thinking_config.thinking_level, "MINIMAL")
        self.assertTrue(request.config.thinking_config.include_thoughts)
        self.assertIsNotNone(request.config.tools[0].google_search.search_types.web_search)
        self.assertIsNotNone(request.config.tools[0].google_search.search_types.image_search)

    def test_reference_image_packing_limits_to_fourteen_images(self):
        class FakeImageArray:
            def __mul__(self, value):
                return self

            def astype(self, dtype):
                return self

        class FakeFrame:
            def cpu(self):
                return self

            def numpy(self):
                return FakeImageArray()

        class FakeTensor:
            def __init__(self, batch_size):
                self.shape = (batch_size, 2, 2, 3)

            def __getitem__(self, index):
                return FakeFrame()

        with mock.patch.object(gemini_service_module.torch, "Tensor", FakeTensor):
            with mock.patch.object(gemini_service_module.Image, "fromarray", side_effect=lambda value: value):
                contents = gemini_service_module.build_contents(
                    prompt="prompt",
                    reference_images=[FakeTensor(20)],
                )

        self.assertEqual(contents[0], "prompt")
        self.assertEqual(len(contents), 15)


class GeminiImageResponseArtifactTests(unittest.TestCase):
    def test_parse_response_preserves_normal_image_mime_and_bytes(self):
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image

        class FakeParsedArray:
            ndim = 3

            def astype(self, dtype):
                return self

            def __truediv__(self, value):
                return self

        tensor = torch.ones((2, 2, 3))
        response = pytypes.SimpleNamespace(
            candidates=[
                pytypes.SimpleNamespace(
                    content=pytypes.SimpleNamespace(
                        parts=[
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(
                                    data=b"jpeg-bytes",
                                    mime_type="image/jpeg",
                                ),
                                thought=False,
                                text=None,
                            )
                        ]
                    )
                )
            ]
        )

        with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
            with mock.patch.object(gemini_service_module.np, "array", return_value=FakeParsedArray(), create=True):
                with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                    parsed = gemini_service_module.parse_generate_content_response(response)

        self.assertEqual(len(parsed.image_artifacts), 1)
        self.assertEqual(parsed.image_artifacts[0].mime_type, "image/jpeg")
        self.assertEqual(parsed.image_artifacts[0].raw_bytes, b"jpeg-bytes")
        self.assertEqual(parsed.image_artifacts[0].source, "response_bytes")

    def test_parse_response_ignores_thinking_images_in_artifacts(self):
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image

        class FakeParsedArray:
            ndim = 3

            def astype(self, dtype):
                return self

            def __truediv__(self, value):
                return self

        tensor = torch.ones((2, 2, 3))
        response = pytypes.SimpleNamespace(
            candidates=[
                pytypes.SimpleNamespace(
                    content=pytypes.SimpleNamespace(
                        parts=[
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(data=b"thought-image", mime_type="image/png"),
                                thought=True,
                                text=None,
                            )
                        ]
                    )
                )
            ]
        )

        with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
            with mock.patch.object(gemini_service_module.np, "array", return_value=FakeParsedArray(), create=True):
                with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                    parsed = gemini_service_module.parse_generate_content_response(response)

        self.assertEqual(parsed.image_artifacts, ())

    def test_parse_response_deduplicates_image_artifacts(self):
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image

        class FakeParsedArray:
            ndim = 3

            def astype(self, dtype):
                return self

            def __truediv__(self, value):
                return self

        tensor = torch.ones((2, 2, 3))
        response = pytypes.SimpleNamespace(
            candidates=[
                pytypes.SimpleNamespace(
                    content=pytypes.SimpleNamespace(
                        parts=[
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(data=b"same-bytes", mime_type="image/webp"),
                                thought=False,
                                text=None,
                            ),
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(data=b"same-bytes", mime_type="image/webp"),
                                thought=False,
                                text=None,
                            ),
                        ]
                    )
                )
            ]
        )

        with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
            with mock.patch.object(gemini_service_module.np, "array", return_value=FakeParsedArray(), create=True):
                with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                    parsed = gemini_service_module.parse_generate_content_response(response)

        self.assertEqual(len(parsed.image_artifacts), 1)
        self.assertEqual(parsed.image_batch.shape[0], 1)


class GeminiImageOutputDefaultTests(unittest.TestCase):
    class _FakeClient:
        def __init__(self, response):
            self.response = response
            self.models = self

        def generate_content(self, model, contents, config):
            return self.response

    def test_generate_image_sets_default_text_outputs_when_only_image_is_returned(self):
        node = BurveGoogleImageGen()
        response = pytypes.SimpleNamespace(
            candidates=[
                pytypes.SimpleNamespace(
                    content=pytypes.SimpleNamespace(
                        parts=[
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(data=b"image-bytes"),
                                thought=False,
                                text=None,
                            )
                        ]
                    ),
                    finish_reason="STOP",
                    safety_ratings=None,
                )
            ],
            prompt_feedback=None,
        )
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image
        class FakeParsedArray:
            ndim = 3

            def astype(self, dtype):
                return self

            def __truediv__(self, value):
                return self

        fake_array = FakeParsedArray()
        tensor = torch.ones((2, 2, 3))

        with mock.patch.object(node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(node, "_build_client", return_value=self._FakeClient(response)):
                with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
                    with mock.patch.object(gemini_service_module.np, "array", return_value=fake_array, create=True):
                        with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                            _, _, thinking_process, system_messages, generated_image_pipe = node.generate_image(
                                prompt="prompt",
                                model={
                                    "model": "gemini-2.5-flash-image",
                                    "aspect_ratio": "1:1",
                                },
                                seed=123,
                            )

        self.assertEqual(thinking_process, gemini_service_module.THINKING_PROCESS_DEFAULT)
        self.assertEqual(system_messages, gemini_service_module.SYSTEM_MESSAGES_DEFAULT)
        self.assertEqual(generated_image_pipe["kind"], nodes_module.GENERATED_IMAGE_PIPE_KIND)


class GeneratedImagePipeTests(unittest.TestCase):
    class _FakeClient:
        def __init__(self, response):
            self.response = response
            self.models = self

        def generate_content(self, model, contents, config):
            return self.response

    def _make_image_response(self, image_bytes, mime_type):
        return pytypes.SimpleNamespace(
            candidates=[
                pytypes.SimpleNamespace(
                    content=pytypes.SimpleNamespace(
                        parts=[
                            pytypes.SimpleNamespace(
                                inline_data=pytypes.SimpleNamespace(data=image_bytes, mime_type=mime_type),
                                thought=False,
                                text=None,
                            )
                        ]
                    ),
                    finish_reason="STOP",
                    safety_ratings=None,
                )
            ],
            prompt_feedback=None,
        )

    def _mock_parse_image(self):
        fake_image = mock.Mock()
        fake_image.convert.return_value = fake_image

        class FakeParsedArray:
            ndim = 3

            def astype(self, dtype):
                return self

            def __truediv__(self, value):
                return self

        tensor = torch.ones((2, 2, 3))
        return fake_image, FakeParsedArray(), tensor

    def test_vertex_generate_image_emits_exact_raw_bytes_in_pipe(self):
        node = BurveVertexImageGen()
        response = self._make_image_response(b"vertex-jpeg", "image/jpeg")
        fake_image, fake_array, tensor = self._mock_parse_image()

        with mock.patch.object(node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(node, "_build_client", return_value=self._FakeClient(response)):
                with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
                    with mock.patch.object(gemini_service_module.np, "array", return_value=fake_array, create=True):
                        with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                            _, _, _, _, generated_image_pipe = node.generate_image(
                                prompt="prompt",
                                model={
                                    "model": "gemini-3.1-flash-image-preview",
                                    "aspect_ratio": "1:1",
                                    "resolution": "1K",
                                    "search_mode": "off",
                                    "thinking_level": "MINIMAL",
                                    "include_thoughts": False,
                                    "output_mime_type": "image/jpeg",
                                    "prominent_people": "allow",
                                },
                                seed=123,
                            )

        self.assertEqual(generated_image_pipe["kind"], GENERATED_IMAGE_PIPE_KIND)
        self.assertEqual(generated_image_pipe["provider_id"], "vertex")
        self.assertEqual(generated_image_pipe["items"][0]["mime_type"], "image/jpeg")
        self.assertEqual(generated_image_pipe["items"][0]["extension"], "jpg")
        self.assertEqual(generated_image_pipe["items"][0]["raw_bytes"], b"vertex-jpeg")

    def test_aistudio_generate_image_emits_pipe_without_raw_bytes(self):
        node = BurveGoogleImageGen()
        response = self._make_image_response(b"aistudio-image", "image/webp")
        fake_image, fake_array, tensor = self._mock_parse_image()

        with mock.patch.object(node, "_get_provider_auth_error", return_value=None):
            with mock.patch.object(node, "_build_client", return_value=self._FakeClient(response)):
                with mock.patch.object(gemini_service_module.Image, "open", return_value=fake_image):
                    with mock.patch.object(gemini_service_module.np, "array", return_value=fake_array, create=True):
                        with mock.patch.object(gemini_service_module.torch, "from_numpy", return_value=tensor):
                            _, _, _, system_messages, generated_image_pipe = node.generate_image(
                                prompt="prompt",
                                model={
                                    "model": "gemini-3.1-flash-image-preview",
                                    "aspect_ratio": "1:1",
                                    "resolution": "1K",
                                    "search_mode": "off",
                                    "thinking_level": "MINIMAL",
                                    "include_thoughts": False,
                                    "output_mime_type": "image/webp",
                                    "prominent_people": "allow",
                                },
                                seed=123,
                            )

        self.assertIn("output_mime_type is not supported", system_messages)
        self.assertEqual(generated_image_pipe["provider_id"], "aistudio")
        self.assertIsNone(generated_image_pipe["items"][0]["raw_bytes"])
        self.assertIn("Exact passthrough bytes are unavailable", generated_image_pipe["notes"])


class BurveSaveGeneratedImageTests(unittest.TestCase):
    def _make_pipe(self, items, notes=""):
        return {
            "kind": GENERATED_IMAGE_PIPE_KIND,
            "version": GENERATED_IMAGE_PIPE_VERSION,
            "provider_id": "vertex",
            "model_id": "gemini-3.1-flash-image-preview",
            "items": items,
            "notes": notes,
        }

    def test_save_generated_image_writes_exact_jpeg_bytes(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((1, 2, 2, 3))
        pipe = self._make_pipe(
            [
                {
                    "mime_type": "image/jpeg",
                    "extension": "jpg",
                    "raw_bytes": b"jpeg-file",
                    "sha256": "hash",
                    "source": "response_bytes",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="vertex", generated_image_pipe=pipe)
                saved_files = output["result"][0]
                saved_path = saved_files.splitlines()[0]
                self.assertTrue(saved_path.endswith(".jpg"))
                with open(saved_path, "rb") as handle:
                    self.assertEqual(handle.read(), b"jpeg-file")
                self.assertEqual(
                    output["ui"]["images"],
                    [{"filename": os.path.basename(saved_path), "subfolder": "", "type": "output"}],
                )

    def test_save_generated_image_falls_back_to_png_without_pipe(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((1, 2, 2, 3))

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="fallback")
                saved_files = output["result"][0]
                saved_path = saved_files.splitlines()[0]
                self.assertTrue(saved_path.endswith(".png"))
                self.assertTrue(os.path.exists(saved_path))
                self.assertEqual(output["ui"]["images"][0]["filename"], os.path.basename(saved_path))
                self.assertEqual(output["ui"]["images"][0]["type"], "output")

    def test_save_generated_image_writes_exact_webp_bytes(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((1, 2, 2, 3))
        pipe = self._make_pipe(
            [
                {
                    "mime_type": "image/webp",
                    "extension": "webp",
                    "raw_bytes": b"webp-file",
                    "sha256": "hash",
                    "source": "response_bytes",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="vertex", generated_image_pipe=pipe)
                saved_files = output["result"][0]
                saved_path = saved_files.splitlines()[0]
                self.assertTrue(saved_path.endswith(".webp"))
                with open(saved_path, "rb") as handle:
                    self.assertEqual(handle.read(), b"webp-file")
                self.assertEqual(output["ui"]["images"][0]["filename"], os.path.basename(saved_path))

    def test_save_generated_image_falls_back_when_raw_bytes_are_missing(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((1, 2, 2, 3))
        pipe = self._make_pipe(
            [
                {
                    "mime_type": "image/webp",
                    "extension": "webp",
                    "raw_bytes": None,
                    "sha256": "hash",
                    "source": "unavailable",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="fallback", generated_image_pipe=pipe)
                saved_files = output["result"][0]

        self.assertIn("raw bytes unavailable, saved PNG fallback", saved_files)
        self.assertTrue(saved_files.splitlines()[0].endswith(".png"))
        self.assertEqual(output["ui"]["images"][0]["type"], "output")

    def test_save_generated_image_falls_back_for_batch_mismatch(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((2, 2, 2, 3))
        pipe = self._make_pipe(
            [
                {
                    "mime_type": "image/jpeg",
                    "extension": "jpg",
                    "raw_bytes": b"jpeg-file",
                    "sha256": "hash",
                    "source": "response_bytes",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="mismatch", generated_image_pipe=pipe)
                saved_files = output["result"][0]

        self.assertIn("item count did not match", saved_files)
        self.assertEqual(len([line for line in saved_files.splitlines() if line.endswith(".png")]), 2)
        self.assertEqual(len(output["ui"]["images"]), 2)

    def test_save_generated_image_rejects_invalid_pipe_kind(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((1, 2, 2, 3))
        pipe = self._make_pipe([])
        pipe["kind"] = "wrong.kind"

        with self.assertRaisesRegex(ValueError, "unexpected kind"):
            node.save(image=image, filename_prefix="invalid", generated_image_pipe=pipe)

    def test_save_generated_image_supports_mixed_batch_outputs(self):
        node = BurveSaveGeneratedImage()
        image = torch.ones((2, 2, 2, 3))
        pipe = self._make_pipe(
            [
                {
                    "mime_type": "image/jpeg",
                    "extension": "jpg",
                    "raw_bytes": b"jpeg-file",
                    "sha256": "hash1",
                    "source": "response_bytes",
                },
                {
                    "mime_type": "image/webp",
                    "extension": "webp",
                    "raw_bytes": None,
                    "sha256": "hash2",
                    "source": "unavailable",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(node, "_get_output_directory", return_value=temp_dir):
                output = node.save(image=image, filename_prefix="mixed", generated_image_pipe=pipe)
                saved_files = output["result"][0]

        saved_lines = saved_files.splitlines()
        self.assertTrue(saved_lines[0].endswith(".jpg"))
        self.assertIn("raw bytes unavailable, saved PNG fallback", saved_files)
        self.assertTrue(any(line.endswith(".png") for line in saved_lines))
        self.assertEqual(len(output["ui"]["images"]), 2)

    def test_save_generated_image_accepts_hidden_prompt_and_extra_pnginfo(self):
        node = BurveSaveGeneratedImage()
        input_types = node.INPUT_TYPES()

        self.assertEqual(
            input_types["hidden"],
            {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        )


class BurveV3ExtensionTests(unittest.TestCase):
    def _get_node_map(self):
        extension = asyncio.run(package_module.comfy_entrypoint())
        node_list = asyncio.run(extension.get_node_list())
        return {node_cls.define_schema().node_id: node_cls for node_cls in node_list}

    def test_package_entrypoint_returns_extension_with_stable_node_ids(self):
        node_map = self._get_node_map()

        self.assertIn("BurveGoogleImageGen", node_map)
        self.assertIn("BurveVertexImageGen", node_map)
        self.assertIn("BurveSaveGeneratedImage", node_map)
        self.assertIn("BurveDebugVertexAuth", node_map)
        self.assertIn("BurveCharacterPlanner", node_map)

    def test_save_generated_image_schema_includes_hidden_prompt_metadata(self):
        node_map = self._get_node_map()
        schema = node_map["BurveSaveGeneratedImage"].define_schema()

        self.assertIn("PROMPT", schema.hidden)
        self.assertIn("EXTRA_PNGINFO", schema.hidden)

    def test_gemini_schema_uses_dynamic_combo_and_removes_legacy_fields(self):
        node_map = self._get_node_map()
        schema = node_map["BurveGoogleImageGen"].define_schema()

        input_ids = [item.id for item in schema.inputs]
        self.assertIn("model", input_ids)
        for legacy_input in (
            "resolution",
            "aspect_ratio",
            "search_mode",
            "thinking_mode",
            "enable_google_search",
            "enable_image_search",
            "enable_thinking_mode",
            "pro_thinking_level",
            "pro_include_thoughts",
        ):
            self.assertNotIn(legacy_input, input_ids)

        model_input = next(item for item in schema.inputs if item.id == "model")
        self.assertEqual(model_input.io_type, "COMFY_DYNAMICCOMBO_V3")

    def test_gemini_schema_keeps_seed_order_and_bool_control_after_generate(self):
        node_map = self._get_node_map()

        for node_id in ("BurveGoogleImageGen", "BurveVertexImageGen"):
            schema = node_map[node_id].define_schema()
            self.assertEqual(
                [item.id for item in schema.inputs],
                [
                    "prompt",
                    "model",
                    "seed",
                    "system_instructions",
                    "reference_images",
                    "character_pipe",
                    "request_timeout_seconds",
                    "retry_attempts",
                ],
            )

            v1_info = schema.get_v1_info(include_hidden=False)
            self.assertEqual(v1_info["input_order"]["required"], ["prompt", "model", "seed"])
            self.assertEqual(
                v1_info["input_order"]["optional"],
                [
                    "system_instructions",
                    "reference_images",
                    "character_pipe",
                    "request_timeout_seconds",
                    "retry_attempts",
                ],
            )
            self.assertIn("seed", v1_info["input"]["required"])
            self.assertIs(v1_info["input"]["required"]["seed"][1]["control_after_generate"], True)
            self.assertNotEqual(v1_info["input"]["required"]["seed"][1]["control_after_generate"], "randomize")
            self.assertTrue(v1_info["input"]["optional"]["request_timeout_seconds"][1]["advanced"])
            self.assertTrue(v1_info["input"]["optional"]["retry_attempts"][1]["advanced"])

    def test_gemini_schema_exposes_generated_image_pipe_output(self):
        node_map = self._get_node_map()

        for node_id in ("BurveGoogleImageGen", "BurveVertexImageGen"):
            schema = node_map[node_id].define_schema()
            self.assertEqual(
                [item.id for item in schema.outputs],
                ["image", "thinking_image", "thinking_process", "system_messages", "generated_image_pipe"],
            )
            self.assertEqual(schema.outputs[-1].io_type, "GENERATED_IMAGE_PIPE")

    def test_dynamic_combo_model_specific_fields_match_specs(self):
        node_map = self._get_node_map()
        schema = node_map["BurveVertexImageGen"].define_schema()
        model_input = next(item for item in schema.inputs if item.id == "model")
        option_map = {option.key: option for option in model_input.options}

        self.assertEqual(
            [item.id for item in option_map["gemini-2.5-flash-image"].inputs],
            ["aspect_ratio"],
        )
        self.assertEqual(
            [item.id for item in option_map["gemini-3-pro-image-preview"].inputs],
            ["aspect_ratio", "resolution", "search_mode"],
        )
        self.assertEqual(
            [item.id for item in option_map["gemini-3.1-flash-image-preview"].inputs],
            [
                "aspect_ratio",
                "resolution",
                "search_mode",
                "thinking_level",
                "include_thoughts",
                "output_mime_type",
                "prominent_people",
            ],
        )

        flash31_fields = {item.id: item for item in option_map["gemini-3.1-flash-image-preview"].inputs}
        self.assertIn("512", flash31_fields["resolution"].options)
        self.assertIn("web+image", flash31_fields["search_mode"].options)
        self.assertIn("image/webp", flash31_fields["output_mime_type"].options)
        self.assertIn("block", flash31_fields["prominent_people"].options)

    def test_non_gemini_wrappers_preserve_node_ids(self):
        node_map = self._get_node_map()

        for node_id in (
            "BurveImageRefPack",
            "BurveImageInfo",
            "BurveSaveGeneratedImage",
            "BurveCharacterPlanner",
            "BurveCharacterRaceDetails",
            "BurveDebugGeminiKey",
            "BurveDebugVertexAuth",
            "BurveSystemInstructions",
            "BurveVariableInjector",
            "BurvePromptDatabase",
            "BurveBlindGridSplitter",
            "BurvePromptSelector14",
        ):
            schema = node_map[node_id].define_schema()
            self.assertEqual(schema.node_id, node_id)

    def test_wrapped_save_node_preserves_ui_payload(self):
        node_map = self._get_node_map()
        save_node = node_map["BurveSaveGeneratedImage"]

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(nodes_module.BurveSaveGeneratedImage, "_get_output_directory", return_value=temp_dir):
                output = save_node.execute(image=torch.ones((1, 2, 2, 3)), filename_prefix="wrapped")

        self.assertIsNotNone(output.ui)
        self.assertIn("images", output.ui)
        self.assertEqual(len(output.ui["images"]), 1)
        self.assertTrue(output.result[0].endswith(".png"))

    def test_wrapped_tuple_result_still_behaves_normally(self):
        node_map = self._get_node_map()
        info_node = node_map["BurveImageInfo"]

        output = info_node.execute(image=torch.ones((1, 2, 2, 3)))

        self.assertIsNone(output.ui)
        self.assertEqual(output.result[1], 2)
        self.assertEqual(output.result[2], 2)

    def test_dependency_gate_fails_when_dynamic_combo_support_is_missing(self):
        v3_extension_module = importlib.import_module("v3_extension")

        with mock.patch.object(sys.modules["comfy_api.latest"].IO, "DynamicCombo", new=pytypes.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "DynamicCombo-capable"):
                asyncio.run(v3_extension_module.comfy_entrypoint())

    def test_dependency_gate_fails_when_google_genai_version_is_too_old(self):
        with mock.patch.object(gemini_service_module.genai, "__version__", "1.52.0"):
            with self.assertRaisesRegex(RuntimeError, "google-genai>=1.68.0,<2"):
                gemini_service_module.ensure_google_genai_compatibility()


class BurveImageInfoTests(unittest.TestCase):
    def test_image_info_reports_dimensions_and_aspect_ratio(self):
        node = BurveImageInfo()

        info, width, height, aspect_ratio = node.inspect(torch.ones((1, 768, 1024, 3)))

        self.assertEqual(width, 1024)
        self.assertEqual(height, 768)
        self.assertEqual(aspect_ratio, "4:3")
        self.assertIn("Size: 1024x768 px", info)
        self.assertIn("Aspect ratio: 4:3", info)

    def test_image_info_reports_square_aspect_ratio(self):
        node = BurveImageInfo()

        _, _, _, aspect_ratio = node.inspect(torch.ones((1, 512, 512, 3)))

        self.assertEqual(aspect_ratio, "1:1")

    def test_image_info_uses_shared_dimensions_for_batched_input(self):
        node = BurveImageInfo()

        _, width, height, aspect_ratio = node.inspect(torch.ones((4, 1080, 1920, 3)))

        self.assertEqual(width, 1920)
        self.assertEqual(height, 1080)
        self.assertEqual(aspect_ratio, "16:9")

    def test_image_info_node_is_registered_in_v3_extension(self):
        extension = asyncio.run(package_module.comfy_entrypoint())
        node_list = asyncio.run(extension.get_node_list())
        node_map = {node_cls.define_schema().node_id: node_cls for node_cls in node_list}

        self.assertIn("BurveImageInfo", node_map)
        self.assertEqual(node_map["BurveImageInfo"].define_schema().node_id, "BurveImageInfo")


class BurveImageGenAuthTests(unittest.TestCase):
    def test_google_node_builds_aistudio_client_with_api_key(self):
        node = BurveGoogleImageGen()

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(nodes_module.genai, "Client") as mock_client:
                node._build_client()

        mock_client.assert_called_once_with(api_key="test-key")

    def test_vertex_node_builds_vertex_client_with_project_and_location(self):
        node = BurveVertexImageGen()

        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "demo-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            },
            clear=False,
        ):
            with mock.patch.object(nodes_module.genai, "Client") as mock_client:
                node._build_client()

        mock_client.assert_called_once_with(
            vertexai=True,
            project="demo-project",
            location="us-central1",
        )

    def test_google_node_missing_auth_mentions_gemini_key_only(self):
        node = BurveGoogleImageGen()

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            error = node._get_provider_auth_error()

        self.assertIn("GEMINI_API_KEY", error)
        self.assertNotIn("GOOGLE_CLOUD_PROJECT", error)
        self.assertNotIn("GOOGLE_CLOUD_LOCATION", error)

    def test_vertex_node_missing_auth_mentions_vertex_setup_not_gemini_key_setup(self):
        node = BurveVertexImageGen()

        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "GEMINI_API_KEY": "should-not-matter",
            },
            clear=False,
        ):
            error = node._get_provider_auth_error()

        self.assertIn("ignores GEMINI_API_KEY", error)
        self.assertIn("GOOGLE_CLOUD_PROJECT", error)
        self.assertIn("GOOGLE_CLOUD_LOCATION", error)
        self.assertIn("gcloud auth application-default login", error)
        self.assertNotIn("setx GEMINI_API_KEY", error)

    def test_vertex_node_explains_that_credentials_do_not_replace_project_and_location(self):
        node = BurveVertexImageGen()

        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/vertex.json",
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "GEMINI_API_KEY": "should-be-ignored",
            },
            clear=False,
        ):
            error = node._get_provider_auth_error()

        self.assertIn("ignores GEMINI_API_KEY", error)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS or ADC authenticates you", error)
        self.assertIn("does not replace GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION", error)
        self.assertIn("Credentials alone do not supply the required project and region", error)
        self.assertIn("Missing required environment variables: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION", error)


class BurveDebugNodeTests(unittest.TestCase):
    def test_debug_gemini_key_reports_presence(self):
        node = BurveDebugGeminiKey()

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "abcd1234efgh"}, clear=False):
            info, = node.run()

        self.assertIn("GEMINI_API_KEY detected inside ComfyUI", info)
        self.assertIn("length = 12", info)
        self.assertIn("start  = abcd", info)
        self.assertIn("end    = efgh", info)

    def test_debug_vertex_auth_reports_env_state(self):
        node = BurveDebugVertexAuth()

        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "demo-project",
                "GOOGLE_CLOUD_LOCATION": "europe-west4",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/vertex.json",
            },
            clear=False,
        ):
            info, = node.run()

        self.assertIn("GOOGLE_CLOUD_PROJECT present: yes", info)
        self.assertIn("GOOGLE_CLOUD_LOCATION present: yes", info)
        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS present: yes", info)
        self.assertIn("project = demo-project", info)
        self.assertIn("location = europe-west4", info)
        self.assertIn("credentials_path = /tmp/vertex.json", info)
        self.assertIn("gcloud auth application-default login", info)

    def test_debug_vertex_auth_explains_credentials_do_not_complete_config(self):
        node = BurveDebugVertexAuth()

        with mock.patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "",
                "GOOGLE_CLOUD_LOCATION": "",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/vertex.json",
            },
            clear=False,
        ):
            info, = node.run()

        self.assertIn("GOOGLE_APPLICATION_CREDENTIALS present: yes", info)
        self.assertIn("Authentication configured: yes", info)
        self.assertIn("Credentials authenticate the request.", info)
        self.assertIn("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION are still required by this node.", info)
        self.assertIn("does not mean the node is fully configured", info)
        self.assertIn("Missing required config: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION", info)


class NodeRegistrationTests(unittest.TestCase):
    def test_package_exports_v3_entrypoint(self):
        self.assertTrue(callable(package_module.comfy_entrypoint))

    def test_package_does_not_export_frontend_web_directory(self):
        self.assertFalse(hasattr(package_module, "WEB_DIRECTORY"))
        self.assertEqual(package_module.__all__, ["comfy_entrypoint"])

    def test_frontend_seed_fix_script_is_removed(self):
        self.assertFalse(os.path.exists(os.path.join(REPO_DIR, "js", "gemini_seed_control_fix.js")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
