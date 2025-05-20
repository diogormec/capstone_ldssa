import pandas as pd

def preparar_features_para_predicao(sku, date, competitor, sales_df_clean, prices_df_clean, campaigns_df_clean, features, model):
    
    """
    Esta função deve replicar a engenharia de features que fizeste no notebook.
    Para já, é um exemplo simples.
    """
    print(f'sku type: {type(sku)}\nsku: {sku}')
    # Extrair informações estruturais do SKU
    sku_subset = sales_df_clean[sales_df_clean['sku'] == sku]
    if sku_subset.empty:
        raise ValueError(f"SKU '{sku}' não encontrado nos dados de vendas.")
    
    sku_info = sku_subset[['structure_level_1', 'structure_level_2', 
                           'structure_level_3', 'structure_level_4']].iloc[0]
    
    # Converter date para datetime se for string
    if isinstance(date, str):
        date = pd.to_datetime(date)
        
    # Criar dataframe para este caso específico com valores padrão
    new_data = {
        'sku': sku,
        'date': date,
        'competitor': competitor,
        'structure_level_1': sku_info['structure_level_1'],
        'structure_level_2': sku_info['structure_level_2'],
        'structure_level_3': sku_info['structure_level_3'],
        'structure_level_4': sku_info['structure_level_4'],
        'year': date.year,
        'month': date.month,
        'day_of_week': date.dayofweek,
        'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
        'is_promo_period': 0,  # Valor padrão
    }
    
    print("Check campains.")
    # Verificar campanhas ativas na data
    active_campaign = 'no_campaign'
    for _, campaign in campaigns_df_clean[campaigns_df_clean['competitor'] == competitor].iterrows():
        if campaign['start_date'] <= date <= campaign['end_date']:
            active_campaign = campaign['chain_campaign']
            break
    
    print("Check active_campaign.")
    # Preencher valores para campanhas
    for feature in features:
        if feature.startswith('campaign_'):
            campaign_name = feature.replace('campaign_', '')
            new_data[feature] = 1 if active_campaign == campaign_name or feature == '' else 0
    
    print("Check leaflets.")
    # Preencher valores para leaflets
    for feature in features:
        if feature.startswith('leaflet_'):
            leaflet_name = feature.replace('leaflet_', '')
            new_data[feature] = 1 if leaflet_name == 'none' else 0

    print(f'Check mean, std, min and max prices for sku.')
    # Para valores estatísticos, usar médias do dataframe completo
    # Esta é uma simplificação; em produção, você armazenaria estas estatísticas
    prices_stats = prices_df_clean[prices_df_clean['sku'] == sku]['target_price']
    if len(prices_stats) > 0:
        new_data['mean_price'] = prices_stats.mean()
        new_data['std_price'] = prices_stats.std() if len(prices_stats) > 1 else 0
        new_data['min_price'] = prices_stats.min()
        new_data['max_price'] = prices_stats.max()

    print(f'Check avg. price by structure code')
    # Médias por nível estrutural
    for level in ['structure_level_1', 'structure_level_2', 'structure_level_3', 'structure_level_4']:
        # Obter os valores únicos do nível atual para o SKU específico
        list_structure_values = sales_df_clean[sales_df_clean['sku'] == sku][level].unique()
        
        # Filtrar preços para SKUs que têm o mesmo valor de estrutura
        filtered_prices = prices_df_clean[prices_df_clean['sku'].isin(list_structure_values)]
        
        # Calcular o preço médio
        avg_price = filtered_prices['target_price'].mean()
        
        # Atribuir o preço médio à coluna correspondente
        new_data[f'{level}_avg_price'] = avg_price
  
    #print('Check competitor_avg_price.')
    # Média de preço do competidor
    '''
    comp_prices = prices_df[prices_df['competitor'] == competitor]['target_price']
    new_data['competitor_avg_price'] = comp_prices.mean() #if len(comp_prices) > 0 else new_data['mean_price']
    #'''

    # Criar dataframe
    df_final = pd.DataFrame([new_data])

    # Selecionar apenas as features usadas no modelo
    available_features = [f for f in features if f in df_final.columns]
    X_pred = df_final[available_features]
    
    # Verificar quais features estão faltando e adicionar valores padrão
    missing_features = set(features) - set(available_features)
    for feature in missing_features:
        if feature.startswith('campaign_'):
            X_pred[feature] = 0
        elif feature.startswith('leaflet_'):
            X_pred[feature] = 0 if feature != 'leaflet_none' else 1
        else:
            # Para outras features, usar a média global
            X_pred[feature] = 0
    
    print("Predicting target_price.")
    
    # Fazer a previsão
    predicted_price = model.predict(X_pred)[0]

    return predicted_price