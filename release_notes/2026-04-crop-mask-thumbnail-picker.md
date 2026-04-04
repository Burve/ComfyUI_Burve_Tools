# April 2026 Crop + Mask Thumbnail Picker Release

## Release Summary

- Added an inline image picker to `Burve Crop + Mask Load` with `List` and `Grid` modes
- Added thumbnail cards for input-folder browsing directly inside the crop editor
- Added inline `Upload` and `Refresh` controls so the hidden native image widget can stay workflow-compatible
- Added a remembered frontend preference at `Burve.CropMaskLoad.ImagePickerMode`
- Added helper JS coverage for filtered image-option matching and selection retention

## Compatibility

- No backend node contract changes
- No workflow migration required
- Existing workflows remain valid because the persisted `image` selection still uses the native widget value
