query_neuromuscularblockers = """
with di as
(
  select
    di.*
    , case
      when itemid = 1052 then 1
      when itemid = 1028 then 1
      when itemid = 1098 then 1
      when itemid = 1858 then 1
      when itemid = 2310 then 1
      when itemid = 2330 then 1
      when itemid = 2360 then 1
      when itemid = 2390 then 1
      when itemid = 2444 then 1
      when itemid = 2463 then 1
      when itemid = 2480 then 1
      when itemid = 2511 then 1
      when itemid = 2517 then 1
      when itemid = 2546 then 1
      when itemid = 30113 then 1
      when itemid = 42385 then 1
      when itemid = 222062 then 1
      when itemid = 227213 then 1
      when itemid = 45096 then 1
      when itemid = 30138 then 1
      when itemid = 1856 then 1
    else 0
    end as neuromuscularblocker
  from d_items di
  where linksto in ('inputevents_mv','inputevents_cv')
)

, mv as
(
  select icustay_id
  , label as neuromuscularblocker_name
  , starttime as neuromuscularblocker_time
  , ROW_NUMBER() over (partition by icustay_id order by starttime, endtime) as rn
  from inputevents_mv mv
  inner join di
      on mv.itemid = di.itemid
      and di.neuromuscularblocker = 1
  where statusdescription != 'Rewritten'
)

, cv as
(
  select icustay_id
  , label as neuromuscularblocker_name
  , charttime as neuromuscularblocker_time
  , ROW_NUMBER() over (partition by icustay_id order by charttime) as rn
  from inputevents_cv cv
  inner join di
      on cv.itemid = di.itemid
      and di.neuromuscularblocker = 1
)

, nm_tbl as
(
select
      ie.icustay_id, ie.hadm_id, ie.intime
    , case when mv.neuromuscularblocker_time is not null and cv.neuromuscularblocker_time is not null
        then case when mv.neuromuscularblocker_time < cv.neuromuscularblocker_time 
                then mv.neuromuscularblocker_name
            else cv.neuromuscularblocker_name
            end
        else coalesce(mv.neuromuscularblocker_name, cv.neuromuscularblocker_name)
     end neuromuscularblocker_name
    , case when mv.neuromuscularblocker_time is not null and cv.neuromuscularblocker_time is not null
        then case when mv.neuromuscularblocker_time < cv.neuromuscularblocker_time 
                then mv.neuromuscularblocker_time
            else cv.neuromuscularblocker_time
            end
        else coalesce(mv.neuromuscularblocker_time, cv.neuromuscularblocker_time)
     end neuromuscularblocker_time
from icustays ie
left join mv
    on ie.icustay_id = mv.icustay_id
left join cv
    on ie.icustay_id = cv.icustay_id
)
"""
