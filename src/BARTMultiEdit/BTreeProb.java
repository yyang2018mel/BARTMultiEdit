package BARTMultiEdit;

import java.util.ArrayList;

public class BTreeProb {

    static double getTreeStructureLogProbability(BTreeNode root, ArrayList<BTreeNode> terminals) {
        var tree_terminals = terminals;
        var tree_internals = root.getNonTerminalsBelowInclusive();

        var sum_terminals_log_prob_split = 0.;
        for(var t : tree_terminals) {
            var log_non_split_prob = 1.-t.calcSplitProbability();
            sum_terminals_log_prob_split += Math.log(log_non_split_prob);
        }

        var sum_nonterminals_log_prob_split = 0.;
        var sum_nonterminals_log_prob_rule = 0.;
        for(var i : tree_internals) {
            var log_split_prob = Math.log(i.calcSplitProbability());
            var log_rule_prob = Math.log(i.calcDecisionProbability());
            sum_nonterminals_log_prob_split += log_split_prob;
            sum_nonterminals_log_prob_rule += log_rule_prob;
        }
        var sum_current_log_probs = sum_nonterminals_log_prob_split + sum_terminals_log_prob_split + sum_nonterminals_log_prob_rule;
        return sum_current_log_probs;
    }

    static double calculateLogGrowProbability(BTreeNode tree_grown_from, BTreeNode grow_node) {
        int b = tree_grown_from.getTerminalsBelowInclusive().size();
        double prob_grow = tree_grown_from.IsStump() ? 1. : tree_grown_from.bartParams.probGrow;
        double p_adj = grow_node.predictorsAvailable.length;
        double n_adj = grow_node.splitsAvailable.get(grow_node.decision.featureIndex).length;
        double log_prob = 4.*Math.log(1.) - Math.log(prob_grow) - Math.log(b) -Math.log(p_adj) - Math.log(n_adj);
        return log_prob;
    }

    static double calculateLogPruneProbability(BTreeNode tree_pruned_from) {
        double prob_prune = tree_pruned_from.IsStump() ? 0. : tree_pruned_from.bartParams.probPrune;
        var num_prunable = tree_pruned_from.getPrunableAndChangeablesBelowInclusive().size();
        double log_prob = 2.*Math.log(1.) - Math.log(prob_prune) - Math.log(num_prunable);
        return log_prob;
    }

    static double calculateLogChangeProbability(BTreeNode tree_changed_from, BTreeNode change_node, double prob_change) {
        var b = tree_changed_from.getPrunableAndChangeablesBelowInclusive().size();
        var p_adj = change_node.predictorsAvailable.length;
        var n_adj = change_node.splitsAvailable.get(change_node.decision.featureIndex).length;
        double log_prob = 4*Math.log(1.) - Math.log(prob_change) - Math.log(b) -Math.log(p_adj) - Math.log(n_adj);
        return log_prob;
    }

    static double getTreeLogLikelihood(BTreeNode root, ArrayList<BTreeNode> terminals, double σ_sq, double σ_mu_sq) {
        var log_likelihood = 0.;
        for(var t : terminals)
            log_likelihood += t.calcTerminalLogLikelihood(σ_sq, σ_mu_sq);
        return log_likelihood;
    }

    static double getTreeStructureLogRatio(BTreeNode proposal_tree, ArrayList<BTreeNode> proposal_terminals,
                                           BTreeNode current_tree, ArrayList<BTreeNode> current_terminals) {
        var current_tree_structure_logprob = getTreeStructureLogProbability(current_tree, current_terminals);
        var proposal_tree_structure_logprob = getTreeStructureLogProbability(proposal_tree, proposal_terminals);
        var ratio = proposal_tree_structure_logprob - current_tree_structure_logprob;

        return ratio;
    }

    static double getTreeLogLikelihoodRatio(BTreeNode proposal_tree, ArrayList<BTreeNode> proposal_terminals,
                                            BTreeNode current_tree, ArrayList<BTreeNode> current_terminals,
                                            double σ_sq, double σ_mu_sq) {
        var current_log_likelihood = getTreeLogLikelihood(current_tree, current_terminals, σ_sq, σ_mu_sq);
        var proposal_log_likelihood = getTreeLogLikelihood(proposal_tree, proposal_terminals, σ_sq, σ_mu_sq);
        var ratio = proposal_log_likelihood - current_log_likelihood;
        return ratio;
    }

}
