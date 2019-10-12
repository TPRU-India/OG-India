VAR_LABELS = {'Y': 'GDP ($Y_t$)', 'C': 'Consumption ($C_t$)',
              'L': 'Labor ($L_t$)', 'G': 'Government Expenditures',
              'TR': 'Lump sum transfers',
              'B': 'Wealth ($B_t$)', 'I_total': 'Investment ($I_t$)',
              'K': 'Capital Stock ($K_t$)',
              'K_d': 'Domestically-owned Capital Stock ($K^d_t$)',
              'K_f': 'Foreign-owned Capital Stock ($K^f_t$)',
              'D': 'Government Debt ($D_t$)',
              'D_d': 'Domestically-owned Gov Debt ($D^d_t$)',
              'D_f': 'Foreign-owned Gov Debt ($D^f_t$)',
              'r': 'Real interest rate ($r_t$)',
              'r_gov': 'Real interest rate on gov debt ($r_{gov,t}$)',
              'r_hh': 'Real interest rate on HH portfolio ($r_{hh,t}$)',
              'w': 'Wage rate', 'BQ': 'Aggregate bequests ($BQ_{j,t}$)',
              'total_revenue': 'Total tax revenue ($REV_t$)',
              'business_revenue': 'Business tax revenue',
              'IITpayroll_revenue': 'IIT and payroll tax revenue',
              'n_mat': 'Labor Supply ($n_{j,s,t}$)',
              'c_path': 'Consumption ($c_{j,s,t}$)',
              'bmat_splus1': 'Savings ($b_{j,s+1,t+1}$)',
              'bq_path': 'Bequests ($bq_{j,s,t}$)',
              'bmat_s': 'Savings ($b_{j,s,t}$)',
              'y_before_tax_mat': 'Before tax income',
              'etr_path': 'Effective Tax Rate ($ETR_{j,s,t}$)',
              'mtrx_path':
              'Marginal Tax Rate, Labor Income ($MTRx_{j,s,t}$)',
              'mtry_path':
              'Marginal Tax Rate, Capital Income ($MTRy_{j,s,t}$)',
              'tax_path':
              'Total Taxes',
              'nssmat': 'Labor Supply ($\\bar{n}_{j,s}$)',
              'bssmat_s': 'Savings ($\\bar{b}_{j,s}$)',
              'bssmat_splus1': 'Savings ($\\bar{n}_{j,s+1}$)',
              'cssmat': 'Consumption ($\\bar{c}_{j,s}$)',
              'yss_before_tax_mat': 'Before-tax Income',
              'etr_ss': 'Effective Tax Rate ($\\bar{ETR}_{j,s}$)',
              'mtrx_ss':
              'Marginal Tax Rate, Labor Income ($\\bar{MTRx}_{j,s}$)',
              'mtry_ss':
              'Marginal Tax Rate, Capital Income ($\\bar{MTRy}_{j,s}$)',
              'ETR': 'Effective Tax Rates',
              'MTRx': 'Marginal Tax Rates on Labor Income',
              'MTRy': 'Marginal Tax Rates on Capital Income'
              }

ToGDP_LABELS = {'D': 'Debt-to-GDP ($D_{t}/Y_t$)',
                'D_d': 'Domestically-owned Debt-to-GDP ($D^d_{t}/Y_t$)',
                'D_f': 'Foreign-owned Debt-to-GDP ($D^f_{t}/Y_t$)',
                'G': 'Govt Spending-to-GDP ($G_{t}/Y_t$)',
                'K': 'Capital-Output Ratio ($K_{t}/Y_t$)',
                'K_d':
                'Domestically-owned Capital-Output Ratio ($K^d_{t}/Y_t$)',
                'K_f':
                'Foreign-owned Capital-Output Ratio ($K^f_{t}/Y_t$)',
                'C': 'Consumption-Output Ratio ($C_{t}/Y_t$)',
                'I': 'Investment-Output Ratio ($I_{t}/Y_t$)',
                'total_revenue': 'Tax Revenue-to-GDP ($REV_{t}/Y_t$)'}

GROUP_LABELS = {0: '0-25%', 1: '25-50%', 2: '50-70%', 3: '70-80%',
                4: '80-90%', 5: '90-99%', 6: 'Top 1%'}
