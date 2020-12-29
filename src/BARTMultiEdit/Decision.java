package BARTMultiEdit;

import java.util.Objects;

class Decision {
    int featureIndex;
    double splitValue;

    Decision(int feature_idx, double split_val) {
        this.featureIndex = feature_idx;
        this.splitValue = split_val;
    }

    @Override
    public String toString() {
        return String.format("(%d,%f)", featureIndex, splitValue);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Decision decision = (Decision) o;
        return featureIndex == decision.featureIndex &&
                Double.compare(decision.splitValue, splitValue) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(featureIndex, splitValue);
    }
}
