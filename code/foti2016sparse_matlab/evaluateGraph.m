% Determines the mean F1 score for the graph based on existence of edges
function [f1_score] = evaluateGraph(inferred_S, true_S)
true_positives = sum(sum((inferred_S == 1) & (true_S == 1)));
false_positives = sum(sum((inferred_S == 1) & (true_S == 0)));
true_negatives = sum(sum((inferred_S == 0) & (true_S == 0)));
false_negatives = sum(sum((inferred_S == 0) & (true_S == 1)));
precision_score_pos = true_positives / (true_positives + false_positives);
recall_score_pos = true_positives / (true_positives + false_negatives);
precision_score_neg = true_negatives / (true_negatives + false_negatives);
recall_score_neg = true_negatives / (true_negatives + false_positives);
f1_score_pos = (2* precision_score_pos * recall_score_pos) / (precision_score_pos + recall_score_pos);
f1_score_neg = (2* precision_score_neg * recall_score_neg) / (precision_score_neg + recall_score_neg);
frac_pos = true_positives / (true_positives + true_negatives); 
frac_neg = 1 - frac_pos;
% f1_score = (f1_score_pos*frac_pos + f1_score_neg*frac_neg) / 2;
f1_score = f1_score_pos;
end