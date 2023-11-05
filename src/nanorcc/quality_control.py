from scipy.stats import pearsonr


class QualityControl:
    def __init__(
        self,
        fov_pct=0.75,
        binding_density=(0.1    , 2.25),
        pos_control_linearity=0.95,
        pos_control_detection_limit=2,
    ):
        self.fov_pct = fov_pct
        self.binding_density = binding_density
        self.pos_control_linearity = pos_control_linearity
        self.pos_control_detection_limit = pos_control_detection_limit

    def fov_qc(self, counts):
        # FOV counts
        fov = counts["FovCounted"] / counts["FovCount"]
        counts["FOV QC"] = fov <= self.fov_pct
        return counts

    def binding_density_qc(self, counts):
        # Binding Density
        counts["Binding Density QC"] = (
            counts["BindingDensity"] <= self.binding_density[0]
        ) | (counts["BindingDensity"] >= self.binding_density[1])
        return counts

    def pos_control_linearity_qc(self, counts):
        # Positive Control Linearity
        pos_cols = counts.filter(regex="POS_").columns
        pos_conc = pos_cols.str.extract("(\d+\.*\d*)").astype(float).values.reshape(-1)
        counts["Positive Control Linearity QC"] = (
            counts[pos_cols].apply(lambda x: pearsonr(x, pos_conc)[0], axis=1) ** 2
            <= self.pos_control_linearity
        )
        return counts

    def pos_control_detection_limit_qc(self, counts):
        # Positive Control Limit of Detection
        mean = counts.filter("^NEG_[A-Z]").mean(axis=1)
        std = counts.filter("^NEG_[A-Z]").std(axis=1)
        counts["Positive Control Detection Limit QC"] = (counts["POS_E(0.5)"]) >= (
            mean + (self.pos_control_detection_limit * std)
        )
        return counts

    def flag_samples(self, counts):
        """Identify samples where QC metrics failed"""
        counts = self.fov_qc(counts)
        counts = self.binding_density_qc(counts)
        counts = self.pos_control_linearity_qc(counts)
        counts = self.pos_control_detection_limit_qc(counts)
        return counts

    def drop_failed_qc(self, counts, reindex=False):
        """Return a dataframe excluding samples that failed QC."""
        qc_cols = [
            "FOV QC",
            "Binding Density QC",
            "Positive Control Linearity QC",
            "Positive Control Detection Limit QC",
        ]
        counts = self.flag_samples(counts).copy()
        counts = counts.loc[~counts[qc_cols].any(axis=1)]
        counts.drop(qc_cols, axis=1, inplace=True)
        if reindex:
            counts.reset_index(drop=True, inplace=True)
        return counts
