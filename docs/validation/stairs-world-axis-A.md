# Stairs world-axis probe A

This is a bounded real MuJoCo v3 probe. It uses no scripted motion or force. The geometry is explicit macro demonstration geometry and is not experimentally supported. Environment friction is 0.8 and coil self-friction is 0.05; support thresholds are unchanged.

- status: `completed`; wall time: `356.297 s`; warning/reset/non-finite: `False`
- segments: `96`; dt: `5e-05`; duration: `2.4`; sampled frames: `121`
- raw movement classification: `sliding`; max penetration: `0.0013037740135000344 m`
- model version: `helical-box-cable-v3`; initial contacts: `0`; initial geometry: `lower material endpoint is positioned near the top tread and upper material endpoint crosses the top tread edge in +x`
- requested/observed world angular velocity: `[0.0, 2.0, 0.0]` / `[8.401837867930705e-13, 2.0000000000000004, 8.239658390640811e-13]`; error norm: `1.1767874475663585e-12`
- candidate flip count: `0` []
- confirmed flip count: `0` []
- verified steps: `[]`; first-touch order: `[(1, 'middle'), (2, 'last'), (3, 'first'), (4, 'first'), (5, 'middle')]`
- energy at final frame: kinetic `0.0005518820639397472`, gravitational `0.014883486275805088`, elastic estimate `0.0007738283066890141`, contact work `-0.17629329352218426`

| step | first touch | first sustained | endpoint support | ambiguous | verified |
|---:|---|---|---|---:|---:|
| 1 | middle @ 0.15734999999999083 | None @ None | None | False | False |
| 2 | last @ 0.33709999999997103 | None @ None | None | False | False |
| 3 | first @ 0.6964499999999315 | first @ 0.746799999999926 | first | False | False |
| 4 | first @ 0.9260499999999062 | middle @ 1.09095000000009 | None | True | False |
| 5 | middle @ 1.1612000000002383 | None @ None | None | False | False |

`summary.json` and `support-diagnostics.json` preserve the substep support transitions and intervals. `frames.json` preserves sampled `contact_details` with contact force, geom IDs, stair index, material index, and tread/other surface classification.
