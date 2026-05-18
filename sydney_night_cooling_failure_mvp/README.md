# Sydney Night-Time Cooling Failure MVP

This project predicts night-time cooling failure risk across selected Sydney suburbs using 2020 ERA5 hourly weather data and static suburb-level urban/geographic features.

## Current study area

The final MVP uses 12 Sydney suburbs across a coastal-to-inland gradient:

- Coastal / water-adjacent: Bondi, Manly, Coogee, Cronulla
- Middle inland: Parramatta, Ryde, Bankstown, Fairfield
- Far inland / western-southwestern: Penrith, Blacktown, Liverpool, Campbelltown

## Target variable

Cooling failure is defined as:

```text
cooling_failure = 1 if:
    day is a hot day
    AND
    overnight cooling rate is in the bottom 25% of hot-day cooling rates
