# March 2026 Vertex AI Launch

## Release Summary

- Added `Burve Google Image Gen (Vertex AI)` as a separate node, so it is obvious which auth path a workflow is using.
- Kept parity with the original AI Studio node by routing both through the same shared generation core.
- Added Vertex-specific auth guidance and the `Burve Debug Vertex Auth` node.
- Expanded the Vertex setup guide in the README.
- Improved `gemini-2.5-flash-image` no-image diagnostics and response handling.

## LinkedIn Post

I shipped a new update for `ComfyUI_Burve_Tools`: `Burve Google Image Gen (Vertex AI)`.

The goal was straightforward. I wanted the same Burve image-generation workflow to work through both auth paths, while keeping it obvious inside ComfyUI whether a workflow is using the regular AI Studio / Gemini API key route or the Vertex AI route. The new node is separate in the graph, but it stays in parity with the original because both nodes now run through the same shared generation core.

One practical reason this matters: Vertex AI gives users access to the Google Cloud free-trial credit path. For people who do not want to rely only on the AI Studio / Gemini API key path, this provides another way to run the same node setup. Google currently advertises this as up to $300 in Google Cloud free-trial credits, subject to Google’s program terms.

I also added `Burve Debug Vertex Auth`, expanded the Vertex setup guide, and improved the node’s diagnostics when Gemini returns a response without a usable image.

If the AI Studio key path is not the route you want to use, try the new Vertex node and check the updated setup section in the repo.

#ComfyUI #GenerativeAI #VertexAI #Gemini #AItools

## Skool Post

New update in `ComfyUI_Burve_Tools`: there is now a separate `Burve Google Image Gen (Vertex AI)` node.

It uses the same shared generation core as the original node, so the workflow stays in sync, but the auth path is now much clearer inside ComfyUI. That means you can use Vertex AI and the Google Cloud free-trial credit path instead of depending only on the AI Studio / Gemini API key route. Google currently advertises that as up to $300 in Google Cloud free-trial credits, subject to their terms.

I also added `Burve Debug Vertex Auth`, expanded the setup guide, and improved no-image diagnostics for `gemini-2.5-flash-image`.

If you want the Vertex route, use the new node and read the Vertex setup section in the README first.

## Key Claims Used

- The repo now includes `Burve Google Image Gen (Vertex AI)`.
- The Vertex node uses standard Vertex auth, not `GEMINI_API_KEY`.
- The Vertex and AI Studio nodes share the same generation core.
- Google Cloud has a free-trial credit program.
- The copy intentionally avoids claiming that AI Studio is universally disabled.
- The copy intentionally frames Vertex as an alternative path for users who want cloud-credit-backed usage.
