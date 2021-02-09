-- Adapted code from:
-- https://github.com/MIT-LCP/mimic-code/blob/master/concepts/pivot/pivoted-lab.sql

DROP TABLE IF EXISTS easylabs CASCADE;
CREATE UNLOGGED TABLE easylabs AS

SELECT subject_id, hadm_id, icustay_id, le.charttime-ch.intime AS icutime,
    AVG(case when itemid in (50882)        and valuenum < 10000 then valuenum end) AS Bicarbonate,
    AVG(case when itemid in (50912)        and valuenum < 150   then valuenum end) AS Creatinine,
    AVG(case when itemid in (50806,50902)  and valuenum < 10000 then valuenum end) AS Chloride,
    AVG(case when itemid in (50809,50931)  and valuenum < 10000 then valuenum end) AS GlucoseLab,
    AVG(case when itemid in (50810,51221)  and valuenum < 100   then valuenum end) AS Hematocrit,
    AVG(case when itemid in (50811,51222)  and valuenum < 50    then valuenum end) AS Hemoglobin,
    AVG(case when itemid in (51265)        and valuenum < 10000 then valuenum end) AS Platelet,
    AVG(case when itemid in (50822, 50971) and valuenum < 30    then valuenum end) AS Potassium,
    AVG(case when itemid in (51275)        and valuenum < 150   then valuenum end) AS PTT,
    AVG(case when itemid in (51237)        and valuenum < 50    then valuenum end) AS INR,
    AVG(case when itemid in (51274)        and valuenum < 150   then valuenum end) AS PT,
    AVG(case when itemid in (50824,50983)  and valuenum < 200   then valuenum end) AS Sodium,
    AVG(case when itemid in (51006)        and valuenum < 300   then valuenum end) AS BUN,
    AVG(case when itemid in (51300,51301)  and valuenum < 1000  then valuenum end) AS WBC,
    AVG(case when itemid in (50862)        and valuenum < 10000 then valuenum end) AS AnionGap,
    AVG(case when itemid in (50868)        and valuenum < 10    then valuenum end) AS Albumin,
    AVG(case when itemid in (51144)        and valuenum < 100   then valuenum end) AS Bands,
    AVG(case when itemid in (50885)        and valuenum < 150   then valuenum end) AS Bilirubin,
    AVG(case when itemid in (50813)        and valuenum < 50    then valuenum end) AS Lactate

FROM cohort ch
INNER JOIN labevents le USING (hadm_id, subject_id)
                        WHERE le.charttime BETWEEN ch.intime AND ch.outtime   -- Only measurements in the ICU
                        AND le.valuenum IS NOT NULL                         -- Not NULL values
                        AND le.valuenum > 0                                 -- Lab values should not be negative
                        AND le.itemid IN                                    -- Prevent empty rows
                            (
                                50862,
                                50868,
                                51144,
                                50882,      
                                50885,
                                50912,      
                                50806,50902,
                                50809,50931,
                                50810,51221,
                                50811,51222,
                                50813,
                                51265,      
                                50822, 50971,
                                51275,      
                                51237,      
                                51274,      
                                50824,50983,
                                51006,      
                                51300,51301
                            )
GROUP BY ch.subject_id, ch.hadm_id, ch.icustay_id, icutime
;
-- Index
CREATE INDEX el_index ON easylabs (icustay_id, icutime)
