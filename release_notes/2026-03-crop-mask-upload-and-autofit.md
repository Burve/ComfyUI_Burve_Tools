# March 2026 Crop Upload + Auto-Fit Release

## Release Summary

- Added drag-and-drop image upload directly inside `Burve Crop + Mask Load`
- Added in-editor refresh/sync of the ComfyUI input image list
- Added V3 translation support for upload/folder/remote image input metadata
- Updated `Burve Crop + Mask Apply` to auto-fit near-match aspect ratios up to `1.0%` estimated trim
- Added advanced `strict_aspect_ratio`
- Added `status` output for warning text
- Added helper JS coverage and tests for synced image selection/upload value handling

## Compatibility

- No dependency changes
- No `requires-comfyui` bump in this release
- Existing workflows remain valid because the new input/output were appended rather than reordered
