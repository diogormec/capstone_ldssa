import pandas as pd

# Main function for preparing features and predicting price
def prepare_features_for_prediction(sku, date, competitor, sales_df_clean, prices_df_clean, campaigns_df_clean, features, model):
    """
    This function replicates the feature engineering done in the notebook.
    """
    print(f'sku type: {type(sku)}\nsku: {sku}')
    
    # Extract structural information for the given SKU
    sku_subset = sales_df_clean[sales_df_clean['sku'] == sku]
    if sku_subset.empty:
        raise ValueError(f"SKU '{sku}' not found in the sales data.")

    sku_info = sku_subset[['structure_level_1', 'structure_level_2', 
                           'structure_level_3', 'structure_level_4']].iloc[0]
    
    # Convert date to datetime if it's a string
    if isinstance(date, str):
        date = pd.to_datetime(date)
        
    # Create a dictionary with default values for the new observation
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
        'is_promo_period': 0,  # Default value, can be customized
    }

    print("Check campaigns.")
    # Determine if there is an active campaign on the given date
    active_campaign = 'no_campaign'
    for _, campaign in campaigns_df_clean[campaigns_df_clean['competitor'] == competitor].iterrows():
        if campaign['start_date'] <= date <= campaign['end_date']:
            active_campaign = campaign['chain_campaign']
            break

    print("Check active_campaign.")
    # Set binary campaign-related features
    for feature in features:
        if feature.startswith('campaign_'):
            campaign_name = feature.replace('campaign_', '')
            new_data[feature] = 1 if active_campaign == campaign_name or feature == '' else 0

    print("Check leaflets.")
    # Set binary leaflet-related features
    for feature in features:
        if feature.startswith('leaflet_'):
            leaflet_name = feature.replace('leaflet_', '')
            new_data[feature] = 1 if leaflet_name == 'none' else 0

    print(f'Check mean, std, min and max prices for sku.')
    # Add statistical price data for the given SKU
    prices_stats = prices_df_clean[prices_df_clean['sku'] == sku]['target_price']
    if len(prices_stats) > 0:
        new_data['mean_price'] = prices_stats.mean()
        new_data['std_price'] = prices_stats.std() if len(prices_stats) > 1 else 0
        new_data['min_price'] = prices_stats.min()
        new_data['max_price'] = prices_stats.max()

    print(f'Check avg. price by structure code')
    # Calculate average prices by structure levels
    for level in ['structure_level_1', 'structure_level_2', 'structure_level_3', 'structure_level_4']:
        # Get unique structure values for this SKU
        list_structure_values = sales_df_clean[sales_df_clean['sku'] == sku][level].unique()
        
        # Filter price records for SKUs that match this structure value
        filtered_prices = prices_df_clean[prices_df_clean['sku'].isin(list_structure_values)]
        
        # Compute and store the average price
        avg_price = filtered_prices['target_price'].mean()
        new_data[f'{level}_avg_price'] = avg_price

    # NOTE: Example of how to add competitor average price (currently commented out)
    '''
    comp_prices = prices_df[prices_df['competitor'] == competitor]['target_price']
    new_data['competitor_avg_price'] = comp_prices.mean()
    '''

    # Create a DataFrame from the constructed data
    df_final = pd.DataFrame([new_data])

    # Filter columns to only those used by the model
    available_features = [f for f in features if f in df_final.columns]
    X_pred = df_final[available_features]
    
    # Fill in missing features with default values
    missing_features = set(features) - set(available_features)
    for feature in missing_features:
        if feature.startswith('campaign_'):
            X_pred[feature] = 0
        elif feature.startswith('leaflet_'):
            X_pred[feature] = 0 if feature != 'leaflet_none' else 1
        else:
            # For any other missing features, assign zero or default value
            X_pred[feature] = 0

    print("Predicting target_price.")
    
    # Make and return the prediction
    predicted_price = model.predict(X_pred)[0]
    return predicted_price
