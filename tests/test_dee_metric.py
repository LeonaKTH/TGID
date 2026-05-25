from procnet.dee.dee_metric import measure_event_table_filling


def test_micro_scores_are_computed_from_counts_without_adjustment():
    event_type_roles = [('EquityFreeze', ['EquityHolder', 'FrozeShares'])]
    event_types = ['EquityFreeze']
    pred = [[[(('company_a',), ('100',))]]]
    gold = [[[(('company_a',), ('200',))]]]

    score = measure_event_table_filling(pred, gold, event_type_roles, event_types)

    assert score['tp'] == 1
    assert score['fp'] == 1
    assert score['fn'] == 1
    assert score['micro_precision'] == 0.5
    assert score['micro_recall'] == 0.5
    assert score['micro_f1'] == 0.5
