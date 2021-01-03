package BARTMultiEdit;

import OpenSourceExtensions.TDoubleHashSetAndArray;

import java.util.*;
import java.util.stream.IntStream;

public class BTreeNode {

    DataContext dataContext;
    Hyperparam bartParams;
    BTreeNode parent; // null if root node
    BTreeNode left, right;

    boolean isTerminal;
    int depth;
    Double mu; // null if non-terminal, otherwise this is the predication value shared by data under this node
    Decision decision;
    int[] dataIndices;
    int[] predictorsAvailable;
    double[] responses;

    /**
     * Return if this is a stump root node
     * */
    boolean IsStump() {
        return this.parent == null && this.left.isTerminal && this.right.isTerminal;
    }

    /**
     * Default constructor
     * */
    BTreeNode() {}

    /**
     * Constructor for a non-terminal, non-root BTreeNode
     * */
    BTreeNode(BTreeNode parent, Decision decision) {
        this.parent = parent;
        this.dataContext = parent.dataContext;
        this.bartParams = parent.bartParams;
        this.decision = decision;
        this.isTerminal = false;
    }

    /**
     * Constructor for a root BTreeNode
     * */
    BTreeNode(DataContext context, Hyperparam bartParams, Decision decision){
        this.dataContext = context;
        this.bartParams = bartParams;
        this.decision = decision;
        this.isTerminal = false;
    }

    /**
     * Constructor for a terminal BTreeNode
     * */
    BTreeNode(BTreeNode parent) {
        this.parent = parent;
        this.dataContext = parent.dataContext;
        this.bartParams = parent.bartParams;
        this.isTerminal = true;
    }

    /**
     * Change a non-terminal node into a terminal node
     * */
    void descendToTerminalNode() {
        this.isTerminal = true;
        this.decision = null;
        this.left = null;
        this.right = null;

    }

    /**
     * Make this BTreeNode data-agnostic
     * */
    void flushBartData() {
        this.dataIndices = null;
        this.predictorsAvailable = null;
        this.responses = null;
        this.dataContext = null;
        if(this.left != null)
            this.left.flushBartData();
        if(this.right != null)
            this.right.flushBartData();
    }

    /**
     * Return the list of non-terminal nodes under this node (inclusive)
     * */
    ArrayList<BTreeNode> getNonTerminalsBelowInclusive() {
        var result = new ArrayList<BTreeNode>();
        if (this.isTerminal) return result;
        var left_result = this.left.getNonTerminalsBelowInclusive();
        var right_result = this.right.getNonTerminalsBelowInclusive();
        result.add(this);
        result.addAll(left_result);
        result.addAll(right_result);
        return result;
    }

    /**
     * Return the list of terminal nodes under this (non-terminal) node
     * */
    ArrayList<BTreeNode> getTerminalsBelowInclusive() {
        var result = new ArrayList<BTreeNode>();
        if (this.isTerminal) {
            result.add(this);
            return result;
        }
        var left_leaves = this.left.getTerminalsBelowInclusive();
        var right_leaves = this.right.getTerminalsBelowInclusive();
        result.addAll(left_leaves);
        result.addAll(right_leaves);
        return result;
    }

    /**
     * Return the list of terminal nodes under this (non-terminal) node with data entry more than n
     * */
    ArrayList<BTreeNode> getTerminalsBelowInclusiveWithDataGEqN(int n) {
        var result = new ArrayList<BTreeNode>();
        if (this.isTerminal) {
            if (this.dataIndices.length >= n)
                result.add(this);
            return result;
        }
        var left_leaves = this.left.getTerminalsBelowInclusiveWithDataGEqN(n);
        var right_leaves = this.right.getTerminalsBelowInclusiveWithDataGEqN(n);
        result.addAll(left_leaves);
        result.addAll(right_leaves);
        return result;
    }

    /**
     * Populate data to each node below (inclusive) and update depth of each node
     * */
    boolean tryPopulateDataAndDepth(double[] responses_from_root) {

        var X = this.dataContext.X;

        if(this.isTerminal && this.dataIndices != null && this.dataIndices.length != 0) {
            if(this.predictorsAvailable == null)
                this.predictorsAvailable = this.getPredictorsThatCouldBeUsedToSplitAtNode()
                                           .stream().mapToInt(i -> i).toArray();
            return true;
        }

        if(this.parent == null) {
            this.depth = 0;
            this.dataIndices = IntStream.range(0, X.length).toArray();
            this.responses = responses_from_root;
        }

        var split_feature_index = decision.featureIndex;
        var split_value = decision.splitValue;
        var left_indices = new ArrayList<Integer>();
        var right_indices = new ArrayList<Integer>();

        for(int idx : this.dataIndices) {
            double[] datum = X[idx];
            if(datum[split_feature_index] <= split_value)
                left_indices.add(idx);
            else
                right_indices.add(idx);
        }

        if(this.predictorsAvailable == null)
            this.predictorsAvailable = this.getPredictorsThatCouldBeUsedToSplitAtNode()
                                       .stream().mapToInt(i -> i).toArray();

        if(left_indices.size() > 0 && right_indices.size() > 0) {
            this.left.dataIndices = left_indices.stream().mapToInt(i -> i).toArray();
            this.left.responses = left_indices.stream().mapToDouble(i -> responses_from_root[i]).toArray();
            this.right.dataIndices = right_indices.stream().mapToInt(i -> i).toArray();
            this.right.responses = right_indices.stream().mapToDouble(i -> responses_from_root[i]).toArray();
            var leftOutcome = this.left.tryPopulateDataAndDepth(responses_from_root);
            var rightOutcome= this.right.tryPopulateDataAndDepth(responses_from_root);
            return leftOutcome && rightOutcome;
        }

        // This tree is not data-worthy. Will be rejected straight away!
        return false;

    }

    boolean tryUpdateResponses(double[] responses_from_root) {

        if(this.isTerminal && this.dataIndices != null && this.dataIndices.length != 0) {
            return true;
        }

        if(this.parent == null)
            this.responses = responses_from_root;

        if(this.left.dataIndices.length > 0 && this.right.dataIndices.length > 0) {
            this.left.responses = Arrays.stream(this.left.dataIndices).mapToDouble(i -> responses_from_root[i]).toArray();
            this.right.responses = Arrays.stream(this.right.dataIndices).mapToDouble(i -> responses_from_root[i]).toArray();
            var leftOutcome = this.left.tryUpdateResponses(responses_from_root);
            var rightOutcome = this.right.tryUpdateResponses(responses_from_root);
            return leftOutcome && rightOutcome;
        }

        return false;

    }

    /**
     * Return the list of prunable/changable nodes below this (non-terminal) node inclusive
     * Namely, non-terminal nodes with both left and right being terminal
     * */
    ArrayList<BTreeNode> getPrunableAndChangeablesBelowInclusive() {
        var result = new ArrayList<BTreeNode>();
        if (this.isTerminal) {
            return result;
        }
        if(this.left.isTerminal && this.right.isTerminal) {
            result.add(this);
            return result;
        }
        var left_prunable_and_changables = this.left.getPrunableAndChangeablesBelowInclusive();
        var right_prunable_and_changables = this.right.getPrunableAndChangeablesBelowInclusive();
        result.addAll(left_prunable_and_changables);
        result.addAll(right_prunable_and_changables);
        return result;
    }

    /**
     * Return the list of features that can be used to split data at this node
     * */
    ArrayList<Integer> getPredictorsThatCouldBeUsedToSplitAtNode() {
        var possible_rule_variables = new ArrayList<Integer>();
        var X_by_col = this.dataContext.XByColumn;
        var num_features = X_by_col.length;
        for (int j = 0; j < num_features; j++){
            //if size of unique of x_i > 1
            double[] x_dot_j = X_by_col[j];
            for (int i = 1; i < dataIndices.length; i++) {
                if (x_dot_j[dataIndices[i - 1]] != x_dot_j[dataIndices[i]]){
                    possible_rule_variables.add(j);
                    break;
                }
            }
        }
        return possible_rule_variables;
    }

    /**
     * Return the list of possible split values for a given feature at this node
     * */
    double[] getPossibleSplitsOfPredictorAtNode(int predictor) {
        double[][] X_by_col = this.dataContext.XByColumn;
        double[] x_dot_j = X_by_col[predictor];
//        double[] x_dot_j_under_node = Arrays.stream(dataIndices).mapToDouble(idx -> x_dot_j[idx]).toArray();
        double[] x_dot_j_under_node = new double[dataIndices.length];
        for(int i = 0; i < dataIndices.length; i++)
            x_dot_j_under_node[i] = x_dot_j[dataIndices[i]];

        TDoubleHashSetAndArray unique_x_dot_j_node = new TDoubleHashSetAndArray(x_dot_j_under_node);
        double max = VectorTools.max(x_dot_j_under_node);
        unique_x_dot_j_node.remove(max); //kill the max
        return unique_x_dot_j_node.toArray();
    }

    /**
     * Do a sampling from the posterior distribution of mu
     * */
    void sampleMu(Random rand, double σ_sq) {
        if (!this.isTerminal) return;
        double posterior_var = this.calcTerminalPosteriorVar(σ_sq);
        double posterior_mean = this.calcTerminalPosteriorMean(σ_sq, posterior_var);
        this.mu = StatToolbox.sample_from_norm_dist(posterior_mean, posterior_var, rand);
    }

    /**
     * Initialize the mu of this node, if a terminal, as 0.
     * */
    void initializeMu() {
        if(!this.isTerminal) return;
        this.mu = 0.;
    }

    /**
     * Return an index of feature that can be used to do splitting at this node
     * */
    int pickRandomPredictorAtNode(Random rand) {
        var predictors_available = this.predictorsAvailable;
        var p_adj = predictors_available.length;
        return predictors_available[(int)Math.floor(rand.nextDouble()*p_adj)];
    }

    /**
     * Return a split value that can be used to do splitting via the given feature at this node
     * */
    double pickRandomSplitValue(Random rand, int feature_index) {
        var split_values = getPossibleSplitsOfPredictorAtNode(feature_index);
        if (split_values.length == 0){
            return Double.NaN;
        }
        int rand_index = (int) Math.floor(rand.nextDouble() * split_values.length);
        return split_values[rand_index];
    }

    /**
     * Return the probability of split at this node
     * */
    double calcSplitProbability() {
        double alpha = this.bartParams.alpha;
        double beta = this.bartParams.beta;
        double prob = alpha/Math.pow(1+depth, beta);
        return prob;
    }

    /**
     * Return the probability of choose this decision at this node
     * */
    double calcDecisionProbability() { // or, rule probability
        if (isTerminal) return Double.NaN;
        var p_adj = this.predictorsAvailable.length;
        var j = decision.featureIndex;
        var nj_adj = this.getPossibleSplitsOfPredictorAtNode(j).length;
        var p_rule = (1. / p_adj) * (1. / nj_adj);
        return p_rule;
    }

    /**
     * Calculate the data likelihood on a terminal node
     * */
    double calcTerminalLogLikelihood(double σ_sq, double σ_mu_sq) {

        if(!this.isTerminal) return Double.NaN;

        var n_eta = this.dataIndices.length;
        var avgResponse = this.avgResponse();
        var response_variation = this.responseVariation();
        var log_likelihood =
                -n_eta/2.*Math.log(2*Math.PI*σ_sq) +
                        0.5*Math.log(σ_sq/(σ_sq+n_eta*σ_mu_sq)) +
                        -1/(2*σ_sq) * (response_variation -
                                Math.pow(avgResponse,2)*Math.pow(n_eta,2)/(n_eta+σ_sq/σ_mu_sq) +
                                n_eta*avgResponse);
        return log_likelihood;
    }

    /**
     * Return the prediction of response value for a given record
     * */
    double getPredictionForData(double[] record) {
        if (this.isTerminal) return this.mu;
        var node_feature_index = this.decision.featureIndex;
        var node_split_value = this.decision.splitValue;
        return ((record[node_feature_index] <= node_split_value)
                ? this.left.getPredictionForData(record)
                : this.right.getPredictionForData(record));
    }

    /**
     * Creates a cloned copy of the tree beginning at this node by recursively cloning its children.
     * */
    public BTreeNode clone() {

        var copy = new BTreeNode();
        copy.dataContext = dataContext;
        copy.bartParams = bartParams;
        copy.parent = parent;
        copy.isTerminal = isTerminal;
        copy.decision = decision;
        copy.dataIndices = dataIndices;
        copy.predictorsAvailable = predictorsAvailable;
        copy.depth = depth;
        copy.mu = mu;
        if(this.left != null) {
            copy.left = this.left.clone();
            copy.left.parent = copy;
        }
        if(this.right != null) {
            copy.right = this.right.clone();
            copy.right.parent = copy;
        }

        return copy;
    }

    private double avgResponse(){
        return StatToolbox.sample_average(this.responses);
    }

    private double responseVariation() { // not variance!
        var avg_response = this.avgResponse();
        var variation = Arrays.stream(this.responses).map(r -> Math.pow((r-avg_response),2)).sum();
        return variation;
    }

    private double calcTerminalPosteriorMean(double σ_sq, double posterior_var) {
        if(!this.isTerminal) return Double.NaN;
        var avg_response = this.avgResponse();
        var hyper_params = this.bartParams;
        var mu_mu = hyper_params.mu_mu;
        var σ_mu_sq = hyper_params.σ_mu_sq;
        var n_eta = this.dataIndices.length;
        var posterior_mean =  (mu_mu / σ_mu_sq + n_eta / σ_sq * avg_response) * posterior_var;
        return posterior_mean;
    }

    private double calcTerminalPosteriorVar(double σ_sq) {
        if(!this.isTerminal) return Double.NaN;
        var σ_mu_sq = this.bartParams.σ_mu_sq;
        var n_eta = this.dataIndices.length;
        var posterior_var = 1 / (1 / σ_mu_sq + n_eta / σ_sq);
        return posterior_var;
    }

    static BTreeNode createStump(Random rand, DataContext context, Hyperparam bartParams) {

        BTreeNode rand_root;
        do {
            var rand_feature_index = rand.nextInt(context.m);
            var rand_split_index = rand.nextInt(context.N);
            var rand_split_value = context.XByColumn[rand_feature_index][rand_split_index];
            var rand_decision = new Decision(rand_feature_index, rand_split_value);
            rand_root = new BTreeNode(context, bartParams, rand_decision);
            rand_root.left = new BTreeNode(rand_root);
            rand_root.right = new BTreeNode(rand_root);
        } while(!rand_root.tryPopulateDataAndDepth(context.yTransformed));

        return rand_root;
    }

    static double[] getInSamplePredictions(BTreeNode root) {
        var pred = new double[root.dataIndices.length];
        root.getTerminalsBelowInclusive().forEach(t -> {
            for(int i = 0; i < t.dataIndices.length; i++)
                pred[t.dataIndices[i]] = t.mu;
        });
        return pred;
    }
}
