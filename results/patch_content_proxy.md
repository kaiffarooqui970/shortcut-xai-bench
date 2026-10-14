# Patch-region content (proxy)

1000 clean test airplanes. Share of images whose patch region is mostly NOT background-coloured, by border-colour similarity (see scripts/patch_content_proxy.py; the proxy overstates overlap for bottom regions because ground under sky counts as content).

| placement | patch | tau = 16 | tau = 24 | tau = 32 |
|---|---|---|---|---|
| configured (corner, inset 1) | obvious (4x4 at 27,27) | 15.9% | 8.0% | 5.0% |
| flush corner (inset 0) | obvious (4x4 at 28,28) | 15.6% | 9.2% | 5.4% |
| origin (22, 22), superseded | obvious (4x4 at 22,22) | 22.2% | 14.7% | 10.0% |
| configured (corner, inset 1) | subtle (8x8 at 23,23) | 14.5% | 6.9% | 4.0% |
| flush corner (inset 0) | subtle (8x8 at 24,24) | 13.2% | 6.3% | 4.0% |
| origin (22, 22), superseded | subtle (8x8 at 22,22) | 14.9% | 8.4% | 3.8% |

Suggested stronger proxy (not yet run): occlusion sensitivity of a model trained on CLEAN data. Occlude the region with the mean colour and measure the drop in airplane probability. That asks directly whether the region holds class evidence, which is what would inflate detection scores.
