query_ventilator = """
SELECT icustay_id,
        DATE_PART('day', icutime) AS ic_t, 
        MAX(OxygenTherapy) AS oxygentherapy,
        MAX(MechVent)  AS mechvent,
        MAX(Extubated) as extubated
FROM easyvent
GROUP BY icustay_id, ic_t
ORDER BY icustay_id, ic_t
    """
