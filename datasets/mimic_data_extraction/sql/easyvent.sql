-- Adapted code from:
-- https://github.com/MIT-LCP/mimic-code/blob/master/concepts/durations/ventilation-durations.sql

DROP TABLE IF EXISTS easyvent CASCADE;
CREATE UNLOGGED TABLE easyvent AS

SELECT icustay_id, ce.charttime-ch.intime AS icutime,
    -- MECHANICAL VENTILATION
    MAX(CASE
        WHEN itemid is NULL or value is NULL                    THEN 0  -- Can't have NULL values
        WHEN itemid = 720       and value != 'Other/Remarks'    THEN 1  -- VentTypeRecorded
        WHEN itemid = 223848    and value != 'Other'            THEN 1
        WHEN itemid = 223849                                    THEN 1  -- Ventilator mode
        WHEN itemid = 467       and value = 'Ventilator'        THEN 1  -- O2 delivery device == ventilator
        WHEN itemid IN
        (445,448,449,450,1340,1486,1600,224687,                     -- Minute volume
        639,654,681,682,683,684,224685,224684,224686,               -- Tidal volume
        218,436,535,444,459,224697,224695,224696,224746,224747,     -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
        221,1,1211,1655,2000,226873,224738,224419,224750,227187,    -- Insp pressure
        543,                                                        -- PlateauPressure
        5865,5866,224707,224709,224705,224706,                      -- APRV pressure
        60,437,505,506,686,220339,224700,                           -- PEEP
        3459,                                                       -- High pressure relief
        501,502,503,224702,                                         -- PCV
        223,667,668,669,670,671,672,                                -- TCPCV
        224701)                                                     -- PSVlevel
                                                                THEN 1
                                                                ELSE 0 end)
        AS MechVent,

    -- OXYGEN THERAPY (I.E. NOT MECHANICAL)
    MAX(CASE
        WHEN itemid = 226732 and value IN
        ('Nasal cannula',               -- 153714 observations
        'Face tent',                    -- 24601 observations
        'Aerosol-cool',                 -- 24560 observations
        'Trach mask ',                  -- 16435 observations
        'High flow neb',                -- 10785 observations
        'Non-rebreather',               -- 5182 observations
        'Venti mask ',                  -- 1947 observations
        'Medium conc mask ',            -- 1888 observations
        'T-piece',                      -- 1135 observations
        'High flow nasal cannula',      -- 925 observations
        'Ultrasonic neb',               -- 9 observations
        'Vapomist')                     -- 3 observations
        THEN 1
        WHEN itemid = 467 and value IN
        ('Cannula',                     -- 278252 observations
        'Nasal Cannula',                -- 248299 observations
        -- 'None',                      -- 95498 observations
        'Face Tent',                    -- 35766 observations
        'Aerosol-Cool',                 -- 33919 observations
        'Trach Mask',                   -- 32655 observations
        'Hi Flow Neb',                  -- 14070 observations
        'Non-Rebreather',               -- 10856 observations
        'Venti Mask',                   -- 4279 observations
        'Medium Conc Mask',             -- 2114 observations
        'Vapotherm',                    -- 1655 observations
        'T-Piece',                      -- 779 observations
        'Hood',                         -- 670 observations
        'Hut',                          -- 150 observations
        'TranstrachealCat',             -- 78 observations
        'Heated Neb',                   -- 37 observations
        'Ultrasonic Neb')               -- 2 observations
        THEN 1
        ELSE 0 end)
        AS OxygenTherapy,

    -- EXTUBATION
    MAX(CASE 
        WHEN itemid is NULL or value is NULL                THEN 0
        WHEN itemid = 640   and value = 'Extubated'         THEN 1
        WHEN itemid = 640   and value = 'Self Extubation'   THEN 1
                                                            ELSE 0 end)
        AS Extubated

FROM cohort ch
INNER JOIN chartevents ce   USING (icustay_id, hadm_id)                         -- Only include WHEN present in both cohort and chartevents
                            WHERE ce.charttime BETWEEN ch.intime AND ch.outtime -- Only measurements in the ICU
                            AND ce.error IS DISTINCT FROM 1                     -- Exclude measurements marked as error
                            AND ce.value IS NOT NULL                            -- Not NULL
                            AND itemid IN
                            (-- Mechanical ventilation
                            720,223849,                                             -- Vent mode
                            223848,                                                 -- Vent type
                            445,448,449,450,1340,1486,1600,224687,                  -- Minute volume
                            639,654,681,682,683,684,224685,224684,224686,           -- Tidal volume
                            218,436,535,444,224697,224695,224696,224746,224747,     -- High/Low/Peak/Mean ("RespPressure")
                            221,1,1211,1655,2000,226873,224738,224419,224750,22718, -- Insp pressure
                            543,                                                    -- PlateauPressure
                            5865,5866,224707,224709,224705,224706,                  -- APRV pressure
                            60,437,505,506,686,220339,224700,                       -- PEEP
                            3459,                                                   -- High pressure relief
                            501,502,503,224702,                                     -- PCV
                            223,667,668,669,670,671,672,                            -- TCPCV
                            224701,                                                 -- PSVlevel
                            -- Extubation
                            640,                                                    -- Extubated
                            -- Oxygen therapy
                            468,226732,                                             -- O2 Delivery Device
                            469,                                                    -- O2 Delivery Mode
                            470,471,223834,                                         -- O2 Flow (lpm)
                            227287,                                                 -- O2 Flow (additional cannula)
                            -- Shared variable
                            467)                                                    -- O2 Delivery Device
GROUP BY icustay_id, icutime

UNION ALL

SELECT icustay_id, starttime-ch.intime AS icutime,
    0 AS MechVent,
    0 AS OxygenTherapy,
    1 AS Extubated
FROM cohort ch 
INNER JOIN procedureevents_mv USING (icustay_id, hadm_id, subject_id)
WHERE starttime BETWEEN ch.intime AND ch.outtime   -- Only measurements in the ICU
AND itemid IN
(227194,    -- "Extubation"
225468,     -- "Unplanned Extubation (patient-initiated)"
225477)     -- "Unplanned Extubation (non-patient initiated)"
ORDER BY icustay_id, icutime
;
-- Index
CREATE INDEX ev_index ON easyvent (icustay_id, icutime)