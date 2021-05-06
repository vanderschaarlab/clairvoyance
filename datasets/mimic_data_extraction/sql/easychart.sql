-- Adapted code from:
-- https://github.com/MIT-LCP/mimic-code/blob/master/concepts/pivot/pivoted-vital.sql

DROP TABLE IF EXISTS easychart CASCADE;
CREATE UNLOGGED TABLE easychart AS

SELECT subject_id, hadm_id, icustay_id, ce.charttime-ch.intime AS icutime,
    -- VITALS 
    AVG(case when itemid in (211,220045)                                and valuenum < 300  then valuenum end)  AS HeartRate,
    AVG(case when itemid in (51,442,455,6701,220179,220050)             and valuenum < 400  then valuenum end)  AS SysBP,
    AVG(case when itemid in (8368,8440,8441,8555,220180,220051)         and valuenum < 300  then valuenum end)  AS DiasBP,
    AVG(case when itemid in (456,52,6702,443,220052,220181,225312)      and valuenum < 300  then valuenum end)  AS MeanBP,
    AVG(case when itemid in (615,618,220210,224690)                     and valuenum < 70   then valuenum end)  AS RespRate,
    AVG(case when itemid in (223761,678)            and valuenum > 70   and valuenum < 120  then (valuenum-32)/1.8
             when itemid in (223762,676)            and valuenum > 10   and valuenum < 50   then valuenum end)  AS Temp,
    AVG(case when itemid in (807,811,1529,3745,3744,225664,220621,226537) and valuenum < 10000  then valuenum end)  AS GlucoseChart,

    -- BLOOD GAS
    AVG(case when itemid in (646,220277)                                and valuenum <= 100 then valuenum end)  AS SpO2,
    AVG(case when itemid in (190,223835)            and valuenum > 0.20 and valuenum <= 1   then valuenum*100
             when itemid in (3420,3422,223835)      and valuenum >= 21  and valuenum <= 100 then valuenum end)  AS FiO2

FROM cohort ch
INNER JOIN chartevents ce   USING (icustay_id, hadm_id, subject_id)
                            WHERE ce.charttime BETWEEN ch.intime AND ch.outtime   -- Only measurements in the ICU
                            AND ce.error IS DISTINCT FROM 1                     -- Exclude measurements marked as error
                            AND ce.valuenum IS NOT NULL                         -- Not NULL
                            AND ce.valuenum > 0                                 -- Lab values should not be negative
                            AND ce.itemid IN                                    -- Prevent empty rows
                                (
                                    211,220045,
                                    51,442,455,6701,220179,220050,
                                    8368,8440,8441,8555,220180,220051,
                                    456,52,6702,443,220052,220181,225312,
                                    615,618,220210,224690,
                                    223761,678,
                                    223762,676,
                                    807,811,1529,3745,3744,225664,220621,226537,
                                    646,220277,
                                    190,223835,
                                    3420,3422,223835,
                                    920,1394,4187,3486,
                                    226730,3485,4188,
                                    762,226512
                                )
GROUP BY ch.subject_id, ch.hadm_id, ch.icustay_id, icutime
;
-- Index
CREATE INDEX ec_index ON easychart (icustay_id, icutime)
