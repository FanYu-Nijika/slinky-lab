# Stairs world-axis probe B

This is a bounded real MuJoCo v3 probe. It uses no scripted motion or force. The geometry is explicit macro demonstration geometry and is not experimentally supported. Environment friction is 0.8 and coil self-friction is 0.05; support thresholds are unchanged.

- status: `completed`; wall time: `376.282 s`; warning/reset/non-finite: `False`
- segments: `96`; dt: `5e-05`; duration: `2.4`; sampled frames: `121`
- raw movement classification: `side_fall`; max penetration: `0.0026535382208822725 m`
- model version: `helical-box-cable-v3`; initial contacts: `0`; initial geometry: `lower material endpoint is positioned near the top tread and upper material endpoint crosses the top tread edge in +x`
- requested/observed world angular velocity: `[0.0, 2.0, 0.0]` / `[8.395391432474769e-13, 2.0000000000000004, 8.262243748772621e-13]`; error norm: `1.1779103904377685e-12`
- candidate flip count: `0` []
- confirmed flip count: `0` []
- verified steps: `[]`; first-touch order: `[(1, 'middle'), (2, 'last'), (3, 'first'), (4, 'middle'), (5, 'middle')]`
- energy at final frame: kinetic `0.00025677822490911486`, gravitational `0.009274225946073755`, elastic estimate `0.004272070150412254`, contact work `-0.3998111977429123`

| step | first touch | first sustained | endpoint support | ambiguous | verified |
|---:|---|---|---|---:|---:|
| 1 | middle @ 0.20954999999998508 | None @ None | None | False | False |
| 2 | last @ 0.5233999999999506 | None @ None | None | False | False |
| 3 | first @ 0.9387999999999048 | None @ None | None | False | False |
| 4 | middle @ 1.1872000000002931 | None @ None | None | False | False |
| 5 | middle @ 1.4559500000008603 | None @ None | None | False | False |

`summary.json` and `support-diagnostics.json` preserve the substep support transitions and intervals. `frames.json` preserves sampled `contact_details` with contact force, geom IDs, stair index, material index, and tread/other surface classification.
