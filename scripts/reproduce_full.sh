#!/bin/bash
# All runs of the report on the full data (850k users), in the order they were made.
# Every stage skips finished artefacts and RL runs resume from checkpoints, so the script
# can be re-run after an interruption. Results go to out_full/.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
D=out_full

# data, SFT, RM-A with all heads and its ensemble, RM-B
$PY run.py --full --stop-after rm

# GRPO / IPO / AWR runs of the main comparison (Table 1)
MAIN=grpo_r1_b2.0_fix,grpo_r3_b2.0_fix,grpo_r6_b1.0_fix,grpo_r7_b2.0_fix,grpo_r7h_b2.0_fix,grpo_r7a_b1.0_fix,grpo_r7a_b2.0_fix,grpo_r8_b0.5_fix,grpo_r8_b1.0_fix,grpo_r8_b2.0_fix,grpo_r8log_b1.0_fix,grpo_r8log_b2.0_fix,grpo_r9_b1.0_fix,grpo_r9_b2.0_fix,ipo_t0.5_fix,awr_fix,awr_hours_g0.5,awr_hours_g1
$PY run.py --full --stop-after rl --runs $MAIN

# offline RL with a critic (section 4.5); critics are trained on first use
CRR=awr_wp50,awr_full,awr_hours_g1_s2,awr_hours_g1_s3,crr_hours_g0.0_exp_b1,crr_hours_g0.0_exp_b1_s2,crr_hours_g0.0_exp_b1_s3,crr_hours_g0.5_exp_b1,crr_hours_g0.9_exp_b1,crr_hours_g0.9_exp_b1_s2,crr_hours_g0.9_exp_b1_s3,crr_hours_g0.9_exp_b0.5,crr_hours_g0.9_exp_b2,crr_hours_g0.9_bin,crr_hours_g0.9_exp_b1_long,crr_wp50_g0.9_exp_b1,crr_wp50_g0.9_bin,awrv_hours_g0.9_b1,awrv_hours_g0.9_b1_s2,awrv_hours_g0.9_b1_s3,awrv_wp50_g0.9_b1
$PY run.py --full --stop-after rl --runs $CRR

# personal reward model RM-P (section 4.6), its calibration report, then GRPO with it
$PY -m rl_recsys.rm_personal --out $D
RMP=grpo_r1_b2.0_p,grpo_r3_b2.0_p,grpo_r3_b1.0_p,grpo_r5_b1.0_fix,grpo_r5_b1.0_p,grpo_r9_b2.0_p,grpo_r9_b1.0_p
$PY run.py --full --stop-after rl --runs $RMP

# test week, 10k users: one table per group (each call writes results_test.csv)
evaluate() { $PY run.py --full --runs $2 --final $2; mv $D/results_test.csv $D/results_test_$1.csv; }
evaluate main $MAIN
evaluate crr $CRR,grpo_r9_b2.0_fix
evaluate rmp $RMP,grpo_r1_b2.0_fix,grpo_r3_b2.0_fix,grpo_r9_b2.0_fix,grpo_r9_b1.0_fix

# analytic optimum, optimisation gap, top-50, within-user RM quality, fresh users
for s in tilted objective_gap eval_at50 within_user_ci eval_fresh; do $PY scripts/$s.py --out $D; done
