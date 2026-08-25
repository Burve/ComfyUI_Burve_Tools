"""Opt-in paid Gemini image smoke tests.

These tests never run in ordinary CI. Set BURVE_RUN_GEMINI_LIVE_TESTS=1 and
provide the normal AI Studio and/or Vertex credentials to enable them.
"""

from __future__ import annotations

import os
import unittest

LIVE_ENABLED = os.getenv("BURVE_RUN_GEMINI_LIVE_TESTS") == "1"

if LIVE_ENABLED:
    from google import genai

    from gemini_image_service import (
        GEMINI_IMAGE_MODEL_SPECS,
        GROUNDING_RENDERER_VERSION,
        build_interaction_pipe,
        clone_config_with_http_timeout,
        parse_generate_content_response,
        parse_interaction_response,
        prepare_gemini_request,
    )


@unittest.skipUnless(
    LIVE_ENABLED,
    "Set BURVE_RUN_GEMINI_LIVE_TESTS=1 to allow paid live Google API calls.",
)
class GeminiImageLiveSmokeTests(unittest.TestCase):
    def _providers(self):
        providers = []
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            providers.append(("aistudio", genai.Client(api_key=api_key), ""))

        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        if project and location:
            providers.append(("vertex", None, location))
        if not providers:
            self.skipTest("No AI Studio or Vertex credentials were configured.")
        return providers

    def _client_for_request(self, provider, base_client, request):
        if provider == "aistudio":
            return base_client
        return genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=request.effective_location,
        )

    def _call(self, client, request):
        if request.api_mode == "interactions":
            response = client.interactions.create(
                **dict(request.interaction_kwargs),
                timeout=120.0,
            )
            return parse_interaction_response(response)
        response = client.models.generate_content(
            model=request.model_id,
            contents=request.contents,
            config=clone_config_with_http_timeout(request.config, 120_000),
        )
        return parse_generate_content_response(response)

    @staticmethod
    def _skip_if_account_restricted(error):
        code = getattr(error, "code", None)
        if code in {400, 403, 404}:
            raise unittest.SkipTest(
                f"Model is unsupported or restricted for this account: {error}"
            )
        raise error

    def test_model_discovery_and_one_low_resolution_generation_per_supported_pair(self):
        for provider, base_client, configured_location in self._providers():
            for model_id in GEMINI_IMAGE_MODEL_SPECS:
                with self.subTest(provider=provider, model=model_id):
                    model = model_id
                    if model_id == "gemini-3.1-flash-image":
                        model = {
                            "model": model_id,
                            "aspect_ratio": "1:1",
                            "resolution": "512",
                            "search_mode": "off",
                            "thinking_level": "MINIMAL",
                            "include_thoughts": False,
                            "output_mime_type": "image/png",
                            "prominent_people": "allow",
                        }
                    request = prepare_gemini_request(
                        provider_id=provider,
                        prompt="Generate a simple blue circle on a white background.",
                        model=model,
                        seed=17,
                        response_mode="image_only",
                        configured_location=configured_location,
                    )
                    if provider == "vertex" and model_id.startswith("gemini-3"):
                        self.assertEqual(request.effective_location, "global")
                    client = self._client_for_request(provider, base_client, request)
                    try:
                        client.models.get(model=model_id)
                        parsed = self._call(client, request)
                    except Exception as error:
                        self._skip_if_account_restricted(error)
                    self.assertGreater(len(parsed.image_artifacts), 0)

    def test_flash_video_image_search_pro_interleaving_and_stateful_edit(self):
        video_path = os.getenv("BURVE_GEMINI_LIVE_VIDEO", "").strip()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            self.skipTest("GEMINI_API_KEY is required for AI Studio Interactions smoke tests.")
        client = genai.Client(api_key=api_key)

        if video_path:
            request = prepare_gemini_request(
                provider_id="aistudio",
                prompt="Create one representative still from this video.",
                model="gemini-3.1-flash-image",
                seed=19,
                video=video_path,
                response_mode="image_only",
            )
            try:
                self.assertGreater(len(self._call(client, request).image_artifacts), 0)
            except Exception as error:
                self._skip_if_account_restricted(error)

        search_model = {
            "model": "gemini-3.1-flash-image",
            "aspect_ratio": "1:1",
            "resolution": "512",
            "search_mode": "image",
            "thinking_level": "MINIMAL",
            "include_thoughts": False,
            "output_mime_type": "image/png",
            "prominent_people": "allow",
        }
        search_request = prepare_gemini_request(
            provider_id="aistudio",
            prompt="Generate an image grounded in current examples of the Google logo.",
            model=search_model,
            seed=23,
            grounding_renderer=GROUNDING_RENDERER_VERSION,
        )
        try:
            grounded = self._call(client, search_request)
        except Exception as error:
            self._skip_if_account_restricted(error)
        self.assertTrue(grounded.search_suggestions_html)

        pro_request = prepare_gemini_request(
            provider_id="aistudio",
            prompt="Explain the composition briefly and generate the image.",
            model="gemini-3-pro-image",
            seed=29,
        )
        try:
            pro = self._call(client, pro_request)
        except Exception as error:
            self._skip_if_account_restricted(error)
        self.assertTrue(pro.response_text)
        self.assertGreater(len(pro.image_artifacts), 0)

        first_request = prepare_gemini_request(
            provider_id="aistudio",
            prompt="Generate a red square on a white background.",
            model="gemini-3.1-flash-image",
            seed=31,
            conversation_mode="stateful",
        )
        try:
            first = self._call(client, first_request)
            pipe = build_interaction_pipe(
                provider_id="aistudio",
                model_id=first_request.model_id,
                api_mode=first_request.api_mode,
                interaction_id=first.interaction_id,
                stored=True,
            )
            second_request = prepare_gemini_request(
                provider_id="aistudio",
                prompt="Change the square to green.",
                model="gemini-3.1-flash-image",
                seed=31,
                conversation_mode="stateful",
                previous_interaction=pipe,
            )
            second = self._call(client, second_request)
        except Exception as error:
            self._skip_if_account_restricted(error)
        self.assertGreater(len(second.image_artifacts), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
