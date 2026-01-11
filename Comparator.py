import ete3
from scipy import stats
import numpy as np

class Comparator:

    @staticmethod
    def RF(tree1, tree2):
        if not isinstance(tree1, ete3.Tree) or not isinstance(tree2, ete3.Tree):
            raise ValueError("trees must be of type ete3.Tree")

        rf, max_rf, common_attrs, p1, p2, d1, d2 = tree1.robinson_foulds(tree2, unrooted_trees=True)
        
        normalized_rf = rf / max_rf if max_rf > 0 else 0
        return normalized_rf

    @staticmethod
    def BSD(tree1, tree2):
        len1 = sum(n.dist for n in tree1.traverse())
        len2 = sum(n.dist for n in tree2.traverse())
        return abs(len1 - len2)

    @staticmethod
    def get_root_to_tip_distances(tree):
        if len(tree.children) > 2:
            tree.set_outgroup(tree.get_midpoint_outgroup())
            
        distances = {}
        for leaf in tree.get_leaves():
            d = tree.get_distance(leaf)
            distances[leaf.name] = d
        return distances

    @staticmethod
    def calculate_temporal_signal(tree, name_to_date_dict):
        distances_map = Comparator.get_root_to_tip_distances(tree)
        
        x_dates = []
        y_distances = []
        
        for name, dist in distances_map.items():
            if name in name_to_date_dict:
                x_dates.append(name_to_date_dict[name])
                y_distances.append(dist)
        
        if len(x_dates) < 3:
            return None

        slope, intercept, r_val, p_val, std_err = stats.linregress(x_dates, y_distances)
        
        return {
            "pearson_r": r_val,
            "p_value": p_val,
            "substitution_rate": slope,
            "r_squared": r_val**2
        }

    @staticmethod
    def root_tip_correlation(tree, dates_dict):
        res = Comparator.calculate_temporal_signal(tree, dates_dict)
        return res["pearson_r"] if res else 0

    @staticmethod
    def root_tip_regression(tree, dates_dict):
        res = Comparator.calculate_temporal_signal(tree, dates_dict)
        return res["substitution_rate"] if res else 0