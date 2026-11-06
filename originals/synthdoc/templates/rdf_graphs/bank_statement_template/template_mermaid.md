```mermaid
graph TD
    %% Node Types
    bankStatement["Bankstatement\n[schema:BankAccount, doco:BodyMatter]"]:::typedNode
    style bankStatement fill:#FFFFFF,stroke:#333
    statement_header["Statement Header\n[doco-fd:Header]"]:::typedNode
    style statement_header fill:#FFD700,stroke:#333
    bank_info_section["Bank Info Section\n[doco-fd:FormSection]"]:::typedNode
    style bank_info_section fill:#F5DEB3,stroke:#333
    bank_name_field["Bank Name Field\n[schema:Organization, doco-fd:FormField]"]:::typedNode
    style bank_name_field fill:#87CEEB,stroke:#333
    bank_address_field["Bank Address Field\n[schema:PostalAddress, doco-fd:FormField]"]:::typedNode
    style bank_address_field fill:#DDA0DD,stroke:#333
    account_info_section["Account Info Section\n[doco-fd:FormSection]"]:::typedNode
    style account_info_section fill:#F5DEB3,stroke:#333
    account_holder_field["Account Holder Field\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style account_holder_field fill:#FFB6C1,stroke:#333
    account_number_field["Account Number Field\n[doco-fd:FormField]"]:::typedNode
    style account_number_field fill:#E6E6FA,stroke:#333
    statement_period_field["Statement Period Field\n[doco-fd:FormField]"]:::typedNode
    style statement_period_field fill:#E6E6FA,stroke:#333
    account_summary_section["Account Summary Section\n[doco-fd:FormSection]"]:::typedNode
    style account_summary_section fill:#F5DEB3,stroke:#333
    beginning_balance_field["Beginning Balance Field\n[schema:MonetaryAmount, doco-fd:FormField]"]:::typedNode
    style beginning_balance_field fill:#90EE90,stroke:#333
    total_deposits_field["Total Deposits Field\n[schema:MonetaryAmount, doco-fd:FormField]"]:::typedNode
    style total_deposits_field fill:#90EE90,stroke:#333
    total_withdrawals_field["Total Withdrawals Field\n[schema:MonetaryAmount, doco-fd:FormField]"]:::typedNode
    style total_withdrawals_field fill:#90EE90,stroke:#333
    ending_balance_field["Ending Balance Field\n[schema:MonetaryAmount, doco-fd:FormField]"]:::typedNode
    style ending_balance_field fill:#90EE90,stroke:#333
    transaction_table["Transaction Table\n[doco:Table]"]:::typedNode
    style transaction_table fill:#FFA07A,stroke:#333
    headerRow["Headerrow\n[doco-fd:TableRow]"]:::typedNode
    style headerRow fill:#FFFFFF,stroke:#333
    transaction_row____loop_index___["Transaction Row {{ Loop.Index }}\n[doco-fd:TableRow]"]:::typedNode
    style transaction_row____loop_index___ fill:#FFFFFF,stroke:#333
    statement_footer["Statement Footer\n[doco-fd:Footer]"]:::typedNode
    style statement_footer fill:#D3D3D3,stroke:#333
    customer_service_field["Customer Service Field\n[doco-fd:FormField]"]:::typedNode
    style customer_service_field fill:#E6E6FA,stroke:#333

    %% Relationships
    value_1753["{{ bank_name }}"]:::valueNode
    bank_name_field -->|name| value_1753
    style value_1753 fill:#f9f,stroke:#666
    value_1753["{{ bank_name }}"]:::valueNode
    bank_name_field -->|value| value_1753
    style value_1753 fill:#f9f,stroke:#666
    value_9369["{{ bank_address }}"]:::valueNode
    bank_address_field -->|address| value_9369
    style value_9369 fill:#f9f,stroke:#666
    value_9369["{{ bank_address }}"]:::valueNode
    bank_address_field -->|value| value_9369
    style value_9369 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    account_holder_field -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_1930["{{ account_holder }}"]:::valueNode
    account_holder_field -->|name| value_1930
    style value_1930 fill:#f9f,stroke:#666
    value_1930["{{ account_holder }}"]:::valueNode
    account_holder_field -->|name| value_1930
    style value_1930 fill:#f9f,stroke:#666
    value_1930["{{ account_holder }}"]:::valueNode
    account_holder_field -->|value| value_1930
    style value_1930 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    account_number_field -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_3144["{{ account_number }}"]:::valueNode
    account_number_field -->|id| value_3144
    style value_3144 fill:#f9f,stroke:#666
    value_3144["{{ account_number }}"]:::valueNode
    account_number_field -->|value| value_3144
    style value_3144 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    statement_period_field -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_7362["{{ start_date }}"]:::valueNode
    statement_period_field -->|startDate| value_7362
    style value_7362 fill:#f9f,stroke:#666
    value_7362["{{ start_date }}"]:::valueNode
    statement_period_field -->|value| value_7362
    style value_7362 fill:#f9f,stroke:#666
    value_2244["{{ end_date }}"]:::valueNode
    statement_period_field -->|endDate| value_2244
    style value_2244 fill:#f9f,stroke:#666
    value_2244["{{ end_date }}"]:::valueNode
    statement_period_field -->|value| value_2244
    style value_2244 fill:#f9f,stroke:#666
    value_2807["Beginning Balance:"]:::valueNode
    beginning_balance_field -->|label| value_2807
    style value_2807 fill:#f9f,stroke:#666
    value_1438["{{ beginning_balance }}"]:::valueNode
    beginning_balance_field -->|amount| value_1438
    style value_1438 fill:#f9f,stroke:#666
    value_1438["{{ beginning_balance }}"]:::valueNode
    beginning_balance_field -->|value| value_1438
    style value_1438 fill:#f9f,stroke:#666
    value_3071["Total Deposits:"]:::valueNode
    total_deposits_field -->|label| value_3071
    style value_3071 fill:#f9f,stroke:#666
    value_8239["{{ total_deposits }}"]:::valueNode
    total_deposits_field -->|amount| value_8239
    style value_8239 fill:#f9f,stroke:#666
    value_8239["{{ total_deposits }}"]:::valueNode
    total_deposits_field -->|value| value_8239
    style value_8239 fill:#f9f,stroke:#666
    value_3556["Total Withdrawals:"]:::valueNode
    total_withdrawals_field -->|label| value_3556
    style value_3556 fill:#f9f,stroke:#666
    value_4752["{{ total_withdrawals }}"]:::valueNode
    total_withdrawals_field -->|amount| value_4752
    style value_4752 fill:#f9f,stroke:#666
    value_4752["{{ total_withdrawals }}"]:::valueNode
    total_withdrawals_field -->|value| value_4752
    style value_4752 fill:#f9f,stroke:#666
    value_15["Ending Balance:"]:::valueNode
    ending_balance_field -->|label| value_15
    style value_15 fill:#f9f,stroke:#666
    value_6728["{{ ending_balance }}"]:::valueNode
    ending_balance_field -->|amount| value_6728
    style value_6728 fill:#f9f,stroke:#666
    value_6728["{{ ending_balance }}"]:::valueNode
    ending_balance_field -->|value| value_6728
    style value_6728 fill:#f9f,stroke:#666
    value_7480["Transaction History"]:::valueNode
    transaction_table -->|caption| value_7480
    style value_7480 fill:#f9f,stroke:#666
    value_9700["Date"]:::valueNode
    headerRow -->|columnHeader| value_9700
    style value_9700 fill:#f9f,stroke:#666
    value_1427["Description"]:::valueNode
    headerRow -->|columnHeader| value_1427
    style value_1427 fill:#f9f,stroke:#666
    value_5759["Withdrawals"]:::valueNode
    headerRow -->|columnHeader| value_5759
    style value_5759 fill:#f9f,stroke:#666
    value_1664["Deposits"]:::valueNode
    headerRow -->|columnHeader| value_1664
    style value_1664 fill:#f9f,stroke:#666
    value_4680["Balance"]:::valueNode
    headerRow -->|columnHeader| value_4680
    style value_4680 fill:#f9f,stroke:#666
    value_5353["{{ transaction.date }}"]:::valueNode
    transaction_row____loop_index___ -->|date| value_5353
    style value_5353 fill:#f9f,stroke:#666
    value_5353["{{ transaction.date }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_5353
    style value_5353 fill:#f9f,stroke:#666
    value_5552["{{ transaction.description }}"]:::valueNode
    transaction_row____loop_index___ -->|name| value_5552
    style value_5552 fill:#f9f,stroke:#666
    value_5552["{{ transaction.description }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_5552
    style value_5552 fill:#f9f,stroke:#666
    value_5552["{{ transaction.description }}"]:::valueNode
    transaction_row____loop_index___ -->|description| value_5552
    style value_5552 fill:#f9f,stroke:#666
    value_5552["{{ transaction.description }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_5552
    style value_5552 fill:#f9f,stroke:#666
    value_590["{{ transaction.ref_code }}"]:::valueNode
    transaction_row____loop_index___ -->|id| value_590
    style value_590 fill:#f9f,stroke:#666
    value_590["{{ transaction.ref_code }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_590
    style value_590 fill:#f9f,stroke:#666
    value_9768["{{ transaction.debit_amt }}"]:::valueNode
    transaction_row____loop_index___ -->|amount| value_9768
    style value_9768 fill:#f9f,stroke:#666
    value_9768["{{ transaction.debit_amt }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_9768
    style value_9768 fill:#f9f,stroke:#666
    value_517["{{ transaction.credit_amt }}"]:::valueNode
    transaction_row____loop_index___ -->|amount| value_517
    style value_517 fill:#f9f,stroke:#666
    value_517["{{ transaction.credit_amt }}"]:::valueNode
    transaction_row____loop_index___ -->|value| value_517
    style value_517 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    transaction_row____loop_index___ -->|amount| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    transaction_row____loop_index___ -->|value| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    customer_service_field -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_6555["{{ customer_service_phone }}"]:::valueNode
    customer_service_field -->|phone| value_6555
    style value_6555 fill:#f9f,stroke:#666
    value_6555["{{ customer_service_phone }}"]:::valueNode
    customer_service_field -->|value| value_6555
    style value_6555 fill:#f9f,stroke:#666
    value_4079["{{ customer_service_email }}"]:::valueNode
    customer_service_field -->|email| value_4079
    style value_4079 fill:#f9f,stroke:#666
    value_4079["{{ customer_service_email }}"]:::valueNode
    customer_service_field -->|value| value_4079
    style value_4079 fill:#f9f,stroke:#666

    %% Hierarchy (containment)
    bankStatement -.->|contains| statement_header
    bankStatement -.->|contains| account_summary_section
    bankStatement -.->|contains| transaction_table
    statement_header -.->|contains| bank_info_section
    statement_header -.->|contains| account_info_section
    bank_info_section -.->|contains| bank_name_field
    bank_info_section -.->|contains| bank_address_field
    account_info_section -.->|contains| account_holder_field
    account_info_section -.->|contains| account_number_field
    account_info_section -.->|contains| statement_period_field
    account_summary_section -.->|contains| beginning_balance_field
    account_summary_section -.->|contains| total_deposits_field
    account_summary_section -.->|contains| total_withdrawals_field
    account_summary_section -.->|contains| ending_balance_field
    transaction_table -.->|contains| headerRow
    headerRow -.->|contains| transaction_row____loop_index___
    transaction_row____loop_index___ -.->|contains| statement_footer
    statement_footer -.->|contains| customer_service_field

    classDef typedNode fill:#fff,stroke:#333,stroke-width:2px
    classDef valueNode fill:#f9f,stroke:#666,stroke-dasharray: 5 5
```
