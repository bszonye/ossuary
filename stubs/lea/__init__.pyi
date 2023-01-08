from .alea import Alea as Alea
from .evidence_ctx import EvidenceCtx as EvidenceCtx
from .ilea import Ilea as Ilia
from .lea import Lea as Lea, P as P, Pf as Pf
from .license import VER as __version__

# make convenient aliases for public static methods of Lea & Alea classes
# all_decreasing = Lea.all_decreasing
# all_different = Lea.all_different
# all_equal = Lea.all_equal
# all_false = Lea.all_false
# all_increasing = Lea.all_increasing
# all_pairwise_verify = Lea.all_pairwise_verify
# all_strict_decreasing = Lea.all_strict_decreasing
# all_strict_increasing = Lea.all_strict_increasing
# all_true = Lea.all_true
# all_verify = Lea.all_verify
# any_false = Lea.any_false
# any_true = Lea.any_true
# any_verify = Lea.any_verify
# bernoulli = Alea.bernoulli
# binom = Lea.binom
# coerce = Alea.coerce
# cpt = Lea.cpt
# dist_l1 = Lea.dist_l1
# dist_l2 = Lea.dist_l2
# event = Alea.event
# evidence = EvidenceCtx
# has_evidence = EvidenceCtx.has_evidence
# add_evidence = EvidenceCtx.add_evidence
# pop_evidence = EvidenceCtx.pop_evidence
# clear_evidence = EvidenceCtx.clear_evidence
# func_wrapper = Lea.func_wrapper
# gen_em_steps = Lea.gen_em_steps
# if_ = Lea.if_
# interval = Alea.interval
# joint = Lea.joint
# learn_by_em = Lea.learn_by_em
# lr = Lea.lr
# mutual_information = Lea.mutual_information
# joint_entropy = Lea.joint_entropy
# make_vars = Lea.make_vars
# max_of = Lea.max_of
# min_of = Lea.min_of
pmf = Alea.pmf
# poisson = Lea.poisson
# read_bif_file = Lea.read_bif_file
# read_csv_file = Alea.read_csv_file
# read_pandas_df = Alea.read_pandas_df
# reduce_all = Lea.reduce_all
set_prob_type = Alea.set_prob_type
vals = Alea.vals
# EXACT = Lea.EXACT
# MCRS = Lea.MCRS
# MCLW = Lea.MCLW
# MCEV = Lea.MCEV
