```mermaid
graph TD
    %% Node Types
    transactionsTable["Transactionstable\n[doco:Table]"]:::typedNode
    style transactionsTable fill:#FFA07A,stroke:#333
    transaction___loop_index___["Transaction{{ Loop.Index }}\n[doco-fd:TableRow]"]:::typedNode
    style transaction___loop_index___ fill:#FFFFFF,stroke:#333
    employmentInfo["Employmentinfo\n[doco-fd:FormSection]"]:::typedNode
    style employmentInfo fill:#F5DEB3,stroke:#333
    employeeName["Employeename\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style employeeName fill:#FFB6C1,stroke:#333
    jobTitle["Jobtitle\n[doco-fd:FormField]"]:::typedNode
    style jobTitle fill:#E6E6FA,stroke:#333
    company["Company\n[schema:Organization, doco-fd:FormField]"]:::typedNode
    style company fill:#87CEEB,stroke:#333
    salary["Salary\n[schema:MonetaryAmount, doco-fd:FormField]"]:::typedNode
    style salary fill:#90EE90,stroke:#333
    ssn["Ssn\n[doco-fd:FormField, doco-fd:TableRow]"]:::typedNode
    style ssn fill:#E6E6FA,stroke:#333
    dob["Dob\n[doco-fd:FormField]"]:::typedNode
    style dob fill:#E6E6FA,stroke:#333
    employeeAddress["Employeeaddress\n[schema:PostalAddress, doco-fd:FormField]"]:::typedNode
    style employeeAddress fill:#DDA0DD,stroke:#333
    personalContact["Personalcontact\n[doco-fd:FormSection]"]:::typedNode
    style personalContact fill:#F5DEB3,stroke:#333
    personName["Personname\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style personName fill:#FFB6C1,stroke:#333
    personPhone["Personphone\n[doco-fd:FormField]"]:::typedNode
    style personPhone fill:#E6E6FA,stroke:#333
    personAddress["Personaddress\n[schema:PostalAddress, doco-fd:FormField]"]:::typedNode
    style personAddress fill:#DDA0DD,stroke:#333
    medicalProvider["Medicalprovider\n[doco-fd:FormSection]"]:::typedNode
    style medicalProvider fill:#F5DEB3,stroke:#333
    provider["Provider\n[schema:Physician, doco-fd:FormField]"]:::typedNode
    style provider fill:#FFFFFF,stroke:#333
    patient["Patient\n[schema:Patient, doco-fd:FormField]"]:::typedNode
    style patient fill:#FFFFFF,stroke:#333
    diagnosis["Diagnosis\n[doco-fd:FormField]"]:::typedNode
    style diagnosis fill:#E6E6FA,stroke:#333
    signatureSection["Signaturesection\n[doco-fd:FormSection]"]:::typedNode
    style signatureSection fill:#F5DEB3,stroke:#333
    attorneySignature["Attorneysignature\n[doco-fd:FormField]"]:::typedNode
    style attorneySignature fill:#E6E6FA,stroke:#333
    signatureDate["Signaturedate\n[doco-fd:FormField]"]:::typedNode
    style signatureDate fill:#E6E6FA,stroke:#333
    notarySection["Notarysection\n[doco-fd:FormSection]"]:::typedNode
    style notarySection fill:#F5DEB3,stroke:#333
    notaryDate["Notarydate\n[doco-fd:FormField]"]:::typedNode
    style notaryDate fill:#E6E6FA,stroke:#333
    notary["Notary\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style notary fill:#FFB6C1,stroke:#333
    commissionNum["Commissionnum\n[doco-fd:FormField]"]:::typedNode
    style commissionNum fill:#E6E6FA,stroke:#333
    witness["Witness\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style witness fill:#FFB6C1,stroke:#333
    invoiceHeader["Invoiceheader\n[doco-fd:Header]"]:::typedNode
    style invoiceHeader fill:#FFD700,stroke:#333
    invoiceNumber["Invoicenumber\n[doco-fd:FormField]"]:::typedNode
    style invoiceNumber fill:#E6E6FA,stroke:#333
    invoiceDate["Invoicedate\n[doco-fd:FormField]"]:::typedNode
    style invoiceDate fill:#E6E6FA,stroke:#333
    dueDate["Duedate\n[doco-fd:FormField]"]:::typedNode
    style dueDate fill:#E6E6FA,stroke:#333
    invoiceCompany["Invoicecompany\n[schema:Organization, doco-fd:FormField]"]:::typedNode
    style invoiceCompany fill:#87CEEB,stroke:#333
    invoiceItemsTable["Invoiceitemstable\n[doco:Table]"]:::typedNode
    style invoiceItemsTable fill:#FFA07A,stroke:#333
    item1["Item1\n[doco-fd:TableRow]"]:::typedNode
    style item1 fill:#FFFFFF,stroke:#333
    item2["Item2\n[doco-fd:TableRow]"]:::typedNode
    style item2 fill:#FFFFFF,stroke:#333
    subtotalRow["Subtotalrow\n[doco-fd:TableRow]"]:::typedNode
    style subtotalRow fill:#FFFFFF,stroke:#333
    taxRow["Taxrow\n[doco-fd:TableRow]"]:::typedNode
    style taxRow fill:#FFFFFF,stroke:#333
    totalRow["Totalrow\n[doco-fd:TableRow]"]:::typedNode
    style totalRow fill:#FFFFFF,stroke:#333
    patient_last_name_field["Patient Last Name Field\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style patient_last_name_field fill:#FFB6C1,stroke:#333
    patient_first_name_field["Patient First Name Field\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style patient_first_name_field fill:#FFB6C1,stroke:#333
    patient_dob_field["Patient Dob Field\n[doco-fd:FormField]"]:::typedNode
    style patient_dob_field fill:#E6E6FA,stroke:#333
    patient_home_address_field["Patient Home Address Field\n[schema:PostalAddress, doco-fd:FormField]"]:::typedNode
    style patient_home_address_field fill:#DDA0DD,stroke:#333
    patient_phone_field["Patient Phone Field\n[doco-fd:FormField]"]:::typedNode
    style patient_phone_field fill:#E6E6FA,stroke:#333
    emergency_contact_field["Emergency Contact Field\n[schema:Person, doco-fd:FormField]"]:::typedNode
    style emergency_contact_field fill:#FFB6C1,stroke:#333
    emergency_relationship_field["Emergency Relationship Field\n[doco-fd:FormField]"]:::typedNode
    style emergency_relationship_field fill:#E6E6FA,stroke:#333
    insurance_provider_field["Insurance Provider Field\n[schema:Organization, doco-fd:FormField]"]:::typedNode
    style insurance_provider_field fill:#87CEEB,stroke:#333
    insurance_policy_field["Insurance Policy Field\n[doco-fd:FormField]"]:::typedNode
    style insurance_policy_field fill:#E6E6FA,stroke:#333
    insurance_group_field["Insurance Group Field\n[doco-fd:FormField]"]:::typedNode
    style insurance_group_field fill:#E6E6FA,stroke:#333
    primary_physician_field["Primary Physician Field\n[schema:Physician, doco-fd:FormField]"]:::typedNode
    style primary_physician_field fill:#FFFFFF,stroke:#333
    medications_field["Medications Field\n[doco-fd:FormField]"]:::typedNode
    style medications_field fill:#E6E6FA,stroke:#333
    visit_reason_field["Visit Reason Field\n[doco-fd:FormField]"]:::typedNode
    style visit_reason_field fill:#E6E6FA,stroke:#333
    earningsTable["Earningstable\n[doco:Table]"]:::typedNode
    style earningsTable fill:#FFA07A,stroke:#333
    regularPay["Regularpay\n[doco-fd:TableRow]"]:::typedNode
    style regularPay fill:#FFFFFF,stroke:#333
    grossPay["Grosspay\n[doco-fd:TableRow]"]:::typedNode
    style grossPay fill:#FFFFFF,stroke:#333
    deductionsTable["Deductionstable\n[doco:Table]"]:::typedNode
    style deductionsTable fill:#FFA07A,stroke:#333
    fedTax["Fedtax\n[doco-fd:TableRow]"]:::typedNode
    style fedTax fill:#FFFFFF,stroke:#333
    stateTax["Statetax\n[doco-fd:TableRow]"]:::typedNode
    style stateTax fill:#FFFFFF,stroke:#333
    medicare["Medicare\n[doco-fd:TableRow]"]:::typedNode
    style medicare fill:#FFFFFF,stroke:#333
    totalDeductions["Totaldeductions\n[doco-fd:TableRow]"]:::typedNode
    style totalDeductions fill:#FFFFFF,stroke:#333
    netPay["Netpay\n[doco-fd:TableRow]"]:::typedNode
    style netPay fill:#FFFFFF,stroke:#333
    loanTermsTable["Loantermstable\n[doco:Table]"]:::typedNode
    style loanTermsTable fill:#FFA07A,stroke:#333
    loanAmount["Loanamount\n[doco-fd:TableRow]"]:::typedNode
    style loanAmount fill:#FFFFFF,stroke:#333
    interestRate["Interestrate\n[doco-fd:TableRow]"]:::typedNode
    style interestRate fill:#FFFFFF,stroke:#333
    monthlyPayment["Monthlypayment\n[doco-fd:TableRow]"]:::typedNode
    style monthlyPayment fill:#FFFFFF,stroke:#333
    closingCosts["Closingcosts\n[doco-fd:TableRow]"]:::typedNode
    style closingCosts fill:#FFFFFF,stroke:#333
    usageSummaryTable["Usagesummarytable\n[doco:Table]"]:::typedNode
    style usageSummaryTable fill:#FFA07A,stroke:#333
    electricService["Electricservice\n[doco-fd:TableRow]"]:::typedNode
    style electricService fill:#FFFFFF,stroke:#333
    gasService["Gasservice\n[doco-fd:TableRow]"]:::typedNode
    style gasService fill:#FFFFFF,stroke:#333
    serviceFees["Servicefees\n[doco-fd:TableRow]"]:::typedNode
    style serviceFees fill:#FFFFFF,stroke:#333
    totalUtility["Totalutility\n[doco-fd:TableRow]"]:::typedNode
    style totalUtility fill:#FFFFFF,stroke:#333
    creditCardTransactions["Creditcardtransactions\n[doco:Table]"]:::typedNode
    style creditCardTransactions fill:#FFA07A,stroke:#333
    ccTrans1["Cctrans1\n[doco-fd:TableRow]"]:::typedNode
    style ccTrans1 fill:#FFFFFF,stroke:#333
    ccTrans2["Cctrans2\n[doco-fd:TableRow]"]:::typedNode
    style ccTrans2 fill:#FFFFFF,stroke:#333
    ccTrans3["Cctrans3\n[doco-fd:TableRow]"]:::typedNode
    style ccTrans3 fill:#FFFFFF,stroke:#333
    ccTrans4["Cctrans4\n[doco-fd:TableRow]"]:::typedNode
    style ccTrans4 fill:#FFFFFF,stroke:#333
    ccTrans5["Cctrans5\n[doco-fd:TableRow]"]:::typedNode
    style ccTrans5 fill:#FFFFFF,stroke:#333
    constructionPaymentApp["Constructionpaymentapp\n[doco:Table]"]:::typedNode
    style constructionPaymentApp fill:#FFA07A,stroke:#333
    constructionItem1["Constructionitem1\n[doco-fd:TableRow]"]:::typedNode
    style constructionItem1 fill:#FFFFFF,stroke:#333
    constructionItem2["Constructionitem2\n[doco-fd:TableRow]"]:::typedNode
    style constructionItem2 fill:#FFFFFF,stroke:#333
    constructionItem3["Constructionitem3\n[doco-fd:TableRow]"]:::typedNode
    style constructionItem3 fill:#FFFFFF,stroke:#333
    constructionTotals["Constructiontotals\n[doco-fd:TableRow]"]:::typedNode
    style constructionTotals fill:#FFFFFF,stroke:#333
    purchaseOrderItems["Purchaseorderitems\n[doco:Table]"]:::typedNode
    style purchaseOrderItems fill:#FFA07A,stroke:#333
    poItem1["Poitem1\n[doco-fd:TableRow]"]:::typedNode
    style poItem1 fill:#FFFFFF,stroke:#333
    poItem2["Poitem2\n[doco-fd:TableRow]"]:::typedNode
    style poItem2 fill:#FFFFFF,stroke:#333
    poItem3["Poitem3\n[doco-fd:TableRow]"]:::typedNode
    style poItem3 fill:#FFFFFF,stroke:#333
    poSubtotal["Posubtotal\n[doco-fd:TableRow]"]:::typedNode
    style poSubtotal fill:#FFFFFF,stroke:#333
    poShipping["Poshipping\n[doco-fd:TableRow]"]:::typedNode
    style poShipping fill:#FFFFFF,stroke:#333
    poTotal["Pototal\n[doco-fd:TableRow]"]:::typedNode
    style poTotal fill:#FFFFFF,stroke:#333

    %% Relationships
    value_7480["Transaction History"]:::valueNode
    transactionsTable -->|caption| value_7480
    style value_7480 fill:#f9f,stroke:#666
    value_9700["Date"]:::valueNode
    transactionsTable -->|columnHeader| value_9700
    style value_9700 fill:#f9f,stroke:#666
    value_1427["Description"]:::valueNode
    transactionsTable -->|columnHeader| value_1427
    style value_1427 fill:#f9f,stroke:#666
    value_9100["Debit"]:::valueNode
    transactionsTable -->|columnHeader| value_9100
    style value_9100 fill:#f9f,stroke:#666
    value_3990["Credit"]:::valueNode
    transactionsTable -->|columnHeader| value_3990
    style value_3990 fill:#f9f,stroke:#666
    value_4680["Balance"]:::valueNode
    transactionsTable -->|columnHeader| value_4680
    style value_4680 fill:#f9f,stroke:#666
    value_5353["{{ transaction.date }}"]:::valueNode
    transaction___loop_index___ -->|cellValue| value_5353
    style value_5353 fill:#f9f,stroke:#666
    value_5353["{{ transaction.date }}"]:::valueNode
    transaction___loop_index___ -->|date| value_5353
    style value_5353 fill:#f9f,stroke:#666
    value_5552["{{ transaction.description }}"]:::valueNode
    transaction___loop_index___ -->|cellValue| value_5552
    style value_5552 fill:#f9f,stroke:#666
    value_2748["{{ transaction.debit }}"]:::valueNode
    transaction___loop_index___ -->|cellValue| value_2748
    style value_2748 fill:#f9f,stroke:#666
    value_2748["{{ transaction.debit }}"]:::valueNode
    transaction___loop_index___ -->|amount| value_2748
    style value_2748 fill:#f9f,stroke:#666
    value_8481["{{ transaction.credit }}"]:::valueNode
    transaction___loop_index___ -->|cellValue| value_8481
    style value_8481 fill:#f9f,stroke:#666
    value_8481["{{ transaction.credit }}"]:::valueNode
    transaction___loop_index___ -->|amount| value_8481
    style value_8481 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    transaction___loop_index___ -->|cellValue| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    transaction___loop_index___ -->|amount| value_0
    style value_0 fill:#f9f,stroke:#666
    value_5777["Employee Name:"]:::valueNode
    employeeName -->|label| value_5777
    style value_5777 fill:#f9f,stroke:#666
    value_5672["{{ employee_name }}"]:::valueNode
    employeeName -->|name| value_5672
    style value_5672 fill:#f9f,stroke:#666
    value_5672["{{ employee_name }}"]:::valueNode
    employeeName -->|value| value_5672
    style value_5672 fill:#f9f,stroke:#666
    value_3585["Job Title:"]:::valueNode
    jobTitle -->|label| value_3585
    style value_3585 fill:#f9f,stroke:#666
    value_9["{{ job_title }}"]:::valueNode
    jobTitle -->|jobTitle| value_9
    style value_9 fill:#f9f,stroke:#666
    value_9["{{ job_title }}"]:::valueNode
    jobTitle -->|value| value_9
    style value_9 fill:#f9f,stroke:#666
    value_2413["Company:"]:::valueNode
    company -->|label| value_2413
    style value_2413 fill:#f9f,stroke:#666
    value_8562["{{ company_name }}"]:::valueNode
    company -->|name| value_8562
    style value_8562 fill:#f9f,stroke:#666
    value_8562["{{ company_name }}"]:::valueNode
    company -->|value| value_8562
    style value_8562 fill:#f9f,stroke:#666
    value_494["Salary:"]:::valueNode
    salary -->|label| value_494
    style value_494 fill:#f9f,stroke:#666
    value_9144["{{ salary_amount }}"]:::valueNode
    salary -->|amount| value_9144
    style value_9144 fill:#f9f,stroke:#666
    value_9144["{{ salary_amount }}"]:::valueNode
    salary -->|value| value_9144
    style value_9144 fill:#f9f,stroke:#666
    value_5397["SSN:"]:::valueNode
    ssn -->|label| value_5397
    style value_5397 fill:#f9f,stroke:#666
    value_9648["{{ ssn }}"]:::valueNode
    ssn -->|id| value_9648
    style value_9648 fill:#f9f,stroke:#666
    value_9648["{{ ssn }}"]:::valueNode
    ssn -->|value| value_9648
    style value_9648 fill:#f9f,stroke:#666
    value_4133["Date of Birth:"]:::valueNode
    dob -->|label| value_4133
    style value_4133 fill:#f9f,stroke:#666
    value_9904["{{ date_of_birth }}"]:::valueNode
    dob -->|birthDate| value_9904
    style value_9904 fill:#f9f,stroke:#666
    value_9904["{{ date_of_birth }}"]:::valueNode
    dob -->|value| value_9904
    style value_9904 fill:#f9f,stroke:#666
    value_6707["Address:"]:::valueNode
    employeeAddress -->|label| value_6707
    style value_6707 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    employeeAddress -->|address| value_5
    style value_5 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    employeeAddress -->|value| value_5
    style value_5 fill:#f9f,stroke:#666
    value_7957["Personal Contact"]:::valueNode
    personalContact -->|sectionTitle| value_7957
    style value_7957 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    personName -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_9151["{{ person_name }}"]:::valueNode
    personName -->|name| value_9151
    style value_9151 fill:#f9f,stroke:#666
    value_9151["{{ person_name }}"]:::valueNode
    personName -->|value| value_9151
    style value_9151 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    personPhone -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    personPhone -->|phone| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    personPhone -->|value| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    personAddress -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    personAddress -->|address| value_5
    style value_5 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    personAddress -->|value| value_5
    style value_5 fill:#f9f,stroke:#666
    value_1573["Medical Provider"]:::valueNode
    medicalProvider -->|sectionTitle| value_1573
    style value_1573 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    provider -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_1683["{{ provider_name }}"]:::valueNode
    provider -->|name| value_1683
    style value_1683 fill:#f9f,stroke:#666
    value_1683["{{ provider_name }}"]:::valueNode
    provider -->|value| value_1683
    style value_1683 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    patient -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_7564["{{ patient_name }}"]:::valueNode
    patient -->|name| value_7564
    style value_7564 fill:#f9f,stroke:#666
    value_7564["{{ patient_name }}"]:::valueNode
    patient -->|value| value_7564
    style value_7564 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    diagnosis -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_1888["{{ diagnosis }}"]:::valueNode
    diagnosis -->|value| value_1888
    style value_1888 fill:#f9f,stroke:#666
    value_9378["{{ plaintiff_name }}"]:::valueNode
    signatureSection -->|name| value_9378
    style value_9378 fill:#f9f,stroke:#666
    value_4071["{{ attorney_name }}"]:::valueNode
    attorneySignature -->|name| value_4071
    style value_4071 fill:#f9f,stroke:#666
    value_4071["{{ attorney_name }}"]:::valueNode
    attorneySignature -->|value| value_4071
    style value_4071 fill:#f9f,stroke:#666
    value_7174["Dated:"]:::valueNode
    signatureDate -->|label| value_7174
    style value_7174 fill:#f9f,stroke:#666
    value_7556["{{ signature_date }}"]:::valueNode
    signatureDate -->|date| value_7556
    style value_7556 fill:#f9f,stroke:#666
    value_7556["{{ signature_date }}"]:::valueNode
    signatureDate -->|value| value_7556
    style value_7556 fill:#f9f,stroke:#666
    value_2004["NOTARY ACKNOWLEDGMENT"]:::valueNode
    notarySection -->|sectionTitle| value_2004
    style value_2004 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    notaryDate -->|date| value_5357
    style value_5357 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    notaryDate -->|value| value_5357
    style value_5357 fill:#f9f,stroke:#666
    value_9339["{{ notary_name }}"]:::valueNode
    notary -->|name| value_9339
    style value_9339 fill:#f9f,stroke:#666
    value_9339["{{ notary_name }}"]:::valueNode
    notary -->|value| value_9339
    style value_9339 fill:#f9f,stroke:#666
    value_5302["Commission No.:"]:::valueNode
    commissionNum -->|label| value_5302
    style value_5302 fill:#f9f,stroke:#666
    value_5167["{{ notary_commission_number }}"]:::valueNode
    commissionNum -->|id| value_5167
    style value_5167 fill:#f9f,stroke:#666
    value_5167["{{ notary_commission_number }}"]:::valueNode
    commissionNum -->|value| value_5167
    style value_5167 fill:#f9f,stroke:#666
    value_5475["Witness:"]:::valueNode
    witness -->|label| value_5475
    style value_5475 fill:#f9f,stroke:#666
    value_5349["{{ witness_name }}"]:::valueNode
    witness -->|name| value_5349
    style value_5349 fill:#f9f,stroke:#666
    value_5349["{{ witness_name }}"]:::valueNode
    witness -->|value| value_5349
    style value_5349 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    invoiceNumber -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_4990["INV-2026-001"]:::valueNode
    invoiceNumber -->|id| value_4990
    style value_4990 fill:#f9f,stroke:#666
    value_4990["INV-2026-001"]:::valueNode
    invoiceNumber -->|value| value_4990
    style value_4990 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    invoiceDate -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    invoiceDate -->|date| value_5357
    style value_5357 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    invoiceDate -->|value| value_5357
    style value_5357 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    dueDate -->|label| value_0
    style value_0 fill:#f9f,stroke:#666
    value_6642["02/04/2026"]:::valueNode
    dueDate -->|paymentDueDate| value_6642
    style value_6642 fill:#f9f,stroke:#666
    value_6642["02/04/2026"]:::valueNode
    dueDate -->|value| value_6642
    style value_6642 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    invoiceCompany -->|name| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    invoiceCompany -->|value| value_0
    style value_0 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    invoiceCompany -->|address| value_5
    style value_5 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    invoiceCompany -->|value| value_5
    style value_5 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    invoiceCompany -->|phone| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    invoiceCompany -->|value| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_6467["Invoice Line Items"]:::valueNode
    invoiceItemsTable -->|caption| value_6467
    style value_6467 fill:#f9f,stroke:#666
    value_8097["Item"]:::valueNode
    invoiceItemsTable -->|columnHeader| value_8097
    style value_8097 fill:#f9f,stroke:#666
    value_1427["Description"]:::valueNode
    invoiceItemsTable -->|columnHeader| value_1427
    style value_1427 fill:#f9f,stroke:#666
    value_3545["Qty"]:::valueNode
    invoiceItemsTable -->|columnHeader| value_3545
    style value_3545 fill:#f9f,stroke:#666
    value_1215["Unit Price"]:::valueNode
    invoiceItemsTable -->|columnHeader| value_1215
    style value_1215 fill:#f9f,stroke:#666
    value_4967["Total"]:::valueNode
    invoiceItemsTable -->|columnHeader| value_4967
    style value_4967 fill:#f9f,stroke:#666
    value_8387["001"]:::valueNode
    item1 -->|cellValue| value_8387
    style value_8387 fill:#f9f,stroke:#666
    value_3177["Professional Consulting Servic..."]:::valueNode
    item1 -->|cellValue| value_3177
    style value_3177 fill:#f9f,stroke:#666
    value_1374["40 hrs"]:::valueNode
    item1 -->|cellValue| value_1374
    style value_1374 fill:#f9f,stroke:#666
    value_8843["$150.00"]:::valueNode
    item1 -->|cellValue| value_8843
    style value_8843 fill:#f9f,stroke:#666
    value_8843["$150.00"]:::valueNode
    item1 -->|amount| value_8843
    style value_8843 fill:#f9f,stroke:#666
    value_9059["$6,000.00"]:::valueNode
    item1 -->|cellValue| value_9059
    style value_9059 fill:#f9f,stroke:#666
    value_9059["$6,000.00"]:::valueNode
    item1 -->|amount| value_9059
    style value_9059 fill:#f9f,stroke:#666
    value_3627["002"]:::valueNode
    item2 -->|cellValue| value_3627
    style value_3627 fill:#f9f,stroke:#666
    value_5123["Technical Documentation"]:::valueNode
    item2 -->|cellValue| value_5123
    style value_5123 fill:#f9f,stroke:#666
    value_6505["1"]:::valueNode
    item2 -->|cellValue| value_6505
    style value_6505 fill:#f9f,stroke:#666
    value_8761["$500.00"]:::valueNode
    item2 -->|cellValue| value_8761
    style value_8761 fill:#f9f,stroke:#666
    value_8761["$500.00"]:::valueNode
    item2 -->|amount| value_8761
    style value_8761 fill:#f9f,stroke:#666
    value_8761["$500.00"]:::valueNode
    item2 -->|cellValue| value_8761
    style value_8761 fill:#f9f,stroke:#666
    value_8761["$500.00"]:::valueNode
    item2 -->|amount| value_8761
    style value_8761 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    subtotalRow -->|rowHeader| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    subtotalRow -->|cellValue| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    subtotalRow -->|amount| value_0
    style value_0 fill:#f9f,stroke:#666
    value_9641["Tax (8.5%):"]:::valueNode
    taxRow -->|rowHeader| value_9641
    style value_9641 fill:#f9f,stroke:#666
    value_5578["$552.50"]:::valueNode
    taxRow -->|cellValue| value_5578
    style value_5578 fill:#f9f,stroke:#666
    value_5578["$552.50"]:::valueNode
    taxRow -->|amount| value_5578
    style value_5578 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    totalRow -->|rowHeader| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    totalRow -->|cellValue| value_0
    style value_0 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    totalRow -->|amount| value_0
    style value_0 fill:#f9f,stroke:#666
    value_3451["{{ ein }}"]:::valueNode
    totalRow -->|value| value_3451
    style value_3451 fill:#f9f,stroke:#666
    value_3451["{{ ein }}"]:::valueNode
    totalRow -->|id| value_3451
    style value_3451 fill:#f9f,stroke:#666
    value_9648["{{ ssn }}"]:::valueNode
    totalRow -->|value| value_9648
    style value_9648 fill:#f9f,stroke:#666
    value_9648["{{ ssn }}"]:::valueNode
    totalRow -->|id| value_9648
    style value_9648 fill:#f9f,stroke:#666
    value_627["2025"]:::valueNode
    totalRow -->|value| value_627
    style value_627 fill:#f9f,stroke:#666
    value_627["2025"]:::valueNode
    totalRow -->|Date| value_627
    style value_627 fill:#f9f,stroke:#666
    value_365["Last Name"]:::valueNode
    patient_last_name_field -->|label| value_365
    style value_365 fill:#f9f,stroke:#666
    value_2106["Rodriguez"]:::valueNode
    patient_last_name_field -->|familyName| value_2106
    style value_2106 fill:#f9f,stroke:#666
    value_2106["Rodriguez"]:::valueNode
    patient_last_name_field -->|value| value_2106
    style value_2106 fill:#f9f,stroke:#666
    value_9420["First Name"]:::valueNode
    patient_first_name_field -->|label| value_9420
    style value_9420 fill:#f9f,stroke:#666
    value_7564["{{ patient_name }}"]:::valueNode
    patient_first_name_field -->|givenName| value_7564
    style value_7564 fill:#f9f,stroke:#666
    value_7564["{{ patient_name }}"]:::valueNode
    patient_first_name_field -->|value| value_7564
    style value_7564 fill:#f9f,stroke:#666
    value_7108["Date of Birth"]:::valueNode
    patient_dob_field -->|label| value_7108
    style value_7108 fill:#f9f,stroke:#666
    value_9904["{{ date_of_birth }}"]:::valueNode
    patient_dob_field -->|birthDate| value_9904
    style value_9904 fill:#f9f,stroke:#666
    value_9904["{{ date_of_birth }}"]:::valueNode
    patient_dob_field -->|value| value_9904
    style value_9904 fill:#f9f,stroke:#666
    value_9277["Home Address"]:::valueNode
    patient_home_address_field -->|label| value_9277
    style value_9277 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    patient_home_address_field -->|address| value_5
    style value_5 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    patient_home_address_field -->|value| value_5
    style value_5 fill:#f9f,stroke:#666
    value_4862["Phone Number"]:::valueNode
    patient_phone_field -->|label| value_4862
    style value_4862 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    patient_phone_field -->|phone| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_3323["{{ phone_number }}"]:::valueNode
    patient_phone_field -->|value| value_3323
    style value_3323 fill:#f9f,stroke:#666
    value_7135["Emergency Contact"]:::valueNode
    emergency_contact_field -->|label| value_7135
    style value_7135 fill:#f9f,stroke:#666
    value_1243["Michael Chen"]:::valueNode
    emergency_contact_field -->|name| value_1243
    style value_1243 fill:#f9f,stroke:#666
    value_1243["Michael Chen"]:::valueNode
    emergency_contact_field -->|value| value_1243
    style value_1243 fill:#f9f,stroke:#666
    value_5104["Relationship"]:::valueNode
    emergency_relationship_field -->|label| value_5104
    style value_5104 fill:#f9f,stroke:#666
    value_9284["Spouse"]:::valueNode
    emergency_relationship_field -->|value| value_9284
    style value_9284 fill:#f9f,stroke:#666
    value_3590["Insurance Provider"]:::valueNode
    insurance_provider_field -->|label| value_3590
    style value_3590 fill:#f9f,stroke:#666
    value_2012["Blue Cross Blue Shield"]:::valueNode
    insurance_provider_field -->|name| value_2012
    style value_2012 fill:#f9f,stroke:#666
    value_2012["Blue Cross Blue Shield"]:::valueNode
    insurance_provider_field -->|value| value_2012
    style value_2012 fill:#f9f,stroke:#666
    value_7397["Policy Number"]:::valueNode
    insurance_policy_field -->|label| value_7397
    style value_7397 fill:#f9f,stroke:#666
    value_5981["BCBS-987654321"]:::valueNode
    insurance_policy_field -->|id| value_5981
    style value_5981 fill:#f9f,stroke:#666
    value_5981["BCBS-987654321"]:::valueNode
    insurance_policy_field -->|value| value_5981
    style value_5981 fill:#f9f,stroke:#666
    value_4518["Group Number"]:::valueNode
    insurance_group_field -->|label| value_4518
    style value_4518 fill:#f9f,stroke:#666
    value_5792["GRP-12345"]:::valueNode
    insurance_group_field -->|id| value_5792
    style value_5792 fill:#f9f,stroke:#666
    value_5792["GRP-12345"]:::valueNode
    insurance_group_field -->|value| value_5792
    style value_5792 fill:#f9f,stroke:#666
    value_1463["Primary Physician"]:::valueNode
    primary_physician_field -->|label| value_1463
    style value_1463 fill:#f9f,stroke:#666
    value_1683["{{ provider_name }}"]:::valueNode
    primary_physician_field -->|name| value_1683
    style value_1683 fill:#f9f,stroke:#666
    value_1683["{{ provider_name }}"]:::valueNode
    primary_physician_field -->|value| value_1683
    style value_1683 fill:#f9f,stroke:#666
    value_7539["Current Medications:"]:::valueNode
    medications_field -->|label| value_7539
    style value_7539 fill:#f9f,stroke:#666
    value_18["Lisinopril 10mg daily, Metform..."]:::valueNode
    medications_field -->|value| value_18
    style value_18 fill:#f9f,stroke:#666
    value_82["Reason for Visit:"]:::valueNode
    visit_reason_field -->|label| value_82
    style value_82 fill:#f9f,stroke:#666
    value_4567["Annual physical examination an..."]:::valueNode
    visit_reason_field -->|value| value_4567
    style value_4567 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    visit_reason_field -->|value| value_5
    style value_5 fill:#f9f,stroke:#666
    value_5["{{ address }}"]:::valueNode
    visit_reason_field -->|address| value_5
    style value_5 fill:#f9f,stroke:#666
    value_1241["12/16/2025"]:::valueNode
    visit_reason_field -->|value| value_1241
    style value_1241 fill:#f9f,stroke:#666
    value_1241["12/16/2025"]:::valueNode
    visit_reason_field -->|Date| value_1241
    style value_1241 fill:#f9f,stroke:#666
    value_3363["12/31/2025"]:::valueNode
    visit_reason_field -->|value| value_3363
    style value_3363 fill:#f9f,stroke:#666
    value_3363["12/31/2025"]:::valueNode
    visit_reason_field -->|Date| value_3363
    style value_3363 fill:#f9f,stroke:#666
    value_7852["01/05/2026"]:::valueNode
    visit_reason_field -->|value| value_7852
    style value_7852 fill:#f9f,stroke:#666
    value_7852["01/05/2026"]:::valueNode
    visit_reason_field -->|Date| value_7852
    style value_7852 fill:#f9f,stroke:#666
    value_5672["{{ employee_name }}"]:::valueNode
    visit_reason_field -->|value| value_5672
    style value_5672 fill:#f9f,stroke:#666
    value_5672["{{ employee_name }}"]:::valueNode
    visit_reason_field -->|name| value_5672
    style value_5672 fill:#f9f,stroke:#666
    value_684["EMP-123456"]:::valueNode
    visit_reason_field -->|value| value_684
    style value_684 fill:#f9f,stroke:#666
    value_684["EMP-123456"]:::valueNode
    visit_reason_field -->|name| value_684
    style value_684 fill:#f9f,stroke:#666
    value_6348["Earnings"]:::valueNode
    earningsTable -->|caption| value_6348
    style value_6348 fill:#f9f,stroke:#666
    value_6348["Earnings"]:::valueNode
    earningsTable -->|columnHeader| value_6348
    style value_6348 fill:#f9f,stroke:#666
    value_9289["Hours"]:::valueNode
    earningsTable -->|columnHeader| value_9289
    style value_9289 fill:#f9f,stroke:#666
    value_7740["Rate"]:::valueNode
    earningsTable -->|columnHeader| value_7740
    style value_7740 fill:#f9f,stroke:#666
    value_4341["Current"]:::valueNode
    earningsTable -->|columnHeader| value_4341
    style value_4341 fill:#f9f,stroke:#666
    value_1977["YTD"]:::valueNode
    earningsTable -->|columnHeader| value_1977
    style value_1977 fill:#f9f,stroke:#666
    value_7433["Regular Pay"]:::valueNode
    regularPay -->|cellValue| value_7433
    style value_7433 fill:#f9f,stroke:#666
    value_4693["80.00"]:::valueNode
    regularPay -->|cellValue| value_4693
    style value_4693 fill:#f9f,stroke:#666
    value_9144["{{ salary_amount }}"]:::valueNode
    regularPay -->|cellValue| value_9144
    style value_9144 fill:#f9f,stroke:#666
    value_9144["{{ salary_amount }}"]:::valueNode
    regularPay -->|amount| value_9144
    style value_9144 fill:#f9f,stroke:#666
    value_3839["$6,800.00"]:::valueNode
    regularPay -->|cellValue| value_3839
    style value_3839 fill:#f9f,stroke:#666
    value_3839["$6,800.00"]:::valueNode
    regularPay -->|amount| value_3839
    style value_3839 fill:#f9f,stroke:#666
    value_8280["$81,600.00"]:::valueNode
    regularPay -->|cellValue| value_8280
    style value_8280 fill:#f9f,stroke:#666
    value_8280["$81,600.00"]:::valueNode
    regularPay -->|amount| value_8280
    style value_8280 fill:#f9f,stroke:#666
    value_9035["Gross Pay"]:::valueNode
    grossPay -->|rowHeader| value_9035
    style value_9035 fill:#f9f,stroke:#666
    value_3839["$6,800.00"]:::valueNode
    grossPay -->|cellValue| value_3839
    style value_3839 fill:#f9f,stroke:#666
    value_3839["$6,800.00"]:::valueNode
    grossPay -->|amount| value_3839
    style value_3839 fill:#f9f,stroke:#666
    value_8280["$81,600.00"]:::valueNode
    grossPay -->|cellValue| value_8280
    style value_8280 fill:#f9f,stroke:#666
    value_8280["$81,600.00"]:::valueNode
    grossPay -->|amount| value_8280
    style value_8280 fill:#f9f,stroke:#666
    value_7605["Deductions"]:::valueNode
    deductionsTable -->|caption| value_7605
    style value_7605 fill:#f9f,stroke:#666
    value_7605["Deductions"]:::valueNode
    deductionsTable -->|columnHeader| value_7605
    style value_7605 fill:#f9f,stroke:#666
    value_4341["Current"]:::valueNode
    deductionsTable -->|columnHeader| value_4341
    style value_4341 fill:#f9f,stroke:#666
    value_1977["YTD"]:::valueNode
    deductionsTable -->|columnHeader| value_1977
    style value_1977 fill:#f9f,stroke:#666
    value_3188["Federal Income Tax"]:::valueNode
    fedTax -->|cellValue| value_3188
    style value_3188 fill:#f9f,stroke:#666
    value_7027["$1,020.00"]:::valueNode
    fedTax -->|cellValue| value_7027
    style value_7027 fill:#f9f,stroke:#666
    value_7027["$1,020.00"]:::valueNode
    fedTax -->|amount| value_7027
    style value_7027 fill:#f9f,stroke:#666
    value_2305["$12,240.00"]:::valueNode
    fedTax -->|cellValue| value_2305
    style value_2305 fill:#f9f,stroke:#666
    value_2305["$12,240.00"]:::valueNode
    fedTax -->|amount| value_2305
    style value_2305 fill:#f9f,stroke:#666
    value_7574["State Income Tax"]:::valueNode
    stateTax -->|cellValue| value_7574
    style value_7574 fill:#f9f,stroke:#666
    value_5652["$340.00"]:::valueNode
    stateTax -->|cellValue| value_5652
    style value_5652 fill:#f9f,stroke:#666
    value_5652["$340.00"]:::valueNode
    stateTax -->|amount| value_5652
    style value_5652 fill:#f9f,stroke:#666
    value_3712["$4,080.00"]:::valueNode
    stateTax -->|cellValue| value_3712
    style value_3712 fill:#f9f,stroke:#666
    value_3712["$4,080.00"]:::valueNode
    stateTax -->|amount| value_3712
    style value_3712 fill:#f9f,stroke:#666
    value_7483["Social Security"]:::valueNode
    ssn -->|cellValue| value_7483
    style value_7483 fill:#f9f,stroke:#666
    value_488["$421.60"]:::valueNode
    ssn -->|cellValue| value_488
    style value_488 fill:#f9f,stroke:#666
    value_488["$421.60"]:::valueNode
    ssn -->|amount| value_488
    style value_488 fill:#f9f,stroke:#666
    value_1840["$5,059.20"]:::valueNode
    ssn -->|cellValue| value_1840
    style value_1840 fill:#f9f,stroke:#666
    value_1840["$5,059.20"]:::valueNode
    ssn -->|amount| value_1840
    style value_1840 fill:#f9f,stroke:#666
    value_485["Medicare"]:::valueNode
    medicare -->|cellValue| value_485
    style value_485 fill:#f9f,stroke:#666
    value_7372["$98.60"]:::valueNode
    medicare -->|cellValue| value_7372
    style value_7372 fill:#f9f,stroke:#666
    value_7372["$98.60"]:::valueNode
    medicare -->|amount| value_7372
    style value_7372 fill:#f9f,stroke:#666
    value_7357["$1,183.20"]:::valueNode
    medicare -->|cellValue| value_7357
    style value_7357 fill:#f9f,stroke:#666
    value_7357["$1,183.20"]:::valueNode
    medicare -->|amount| value_7357
    style value_7357 fill:#f9f,stroke:#666
    value_3498["Total Deductions"]:::valueNode
    totalDeductions -->|rowHeader| value_3498
    style value_3498 fill:#f9f,stroke:#666
    value_1497["$1,880.20"]:::valueNode
    totalDeductions -->|cellValue| value_1497
    style value_1497 fill:#f9f,stroke:#666
    value_1497["$1,880.20"]:::valueNode
    totalDeductions -->|amount| value_1497
    style value_1497 fill:#f9f,stroke:#666
    value_6824["$22,562.40"]:::valueNode
    totalDeductions -->|cellValue| value_6824
    style value_6824 fill:#f9f,stroke:#666
    value_6824["$22,562.40"]:::valueNode
    totalDeductions -->|amount| value_6824
    style value_6824 fill:#f9f,stroke:#666
    value_8619["NET PAY"]:::valueNode
    netPay -->|rowHeader| value_8619
    style value_8619 fill:#f9f,stroke:#666
    value_7871["$4,919.80"]:::valueNode
    netPay -->|cellValue| value_7871
    style value_7871 fill:#f9f,stroke:#666
    value_7871["$4,919.80"]:::valueNode
    netPay -->|amount| value_7871
    style value_7871 fill:#f9f,stroke:#666
    value_356["$59,037.60"]:::valueNode
    netPay -->|cellValue| value_356
    style value_356 fill:#f9f,stroke:#666
    value_356["$59,037.60"]:::valueNode
    netPay -->|amount| value_356
    style value_356 fill:#f9f,stroke:#666
    value_7553["$850,000.00"]:::valueNode
    netPay -->|value| value_7553
    style value_7553 fill:#f9f,stroke:#666
    value_7553["$850,000.00"]:::valueNode
    netPay -->|MonetaryAmount| value_7553
    style value_7553 fill:#f9f,stroke:#666
    value_1576["Loan Terms"]:::valueNode
    loanTermsTable -->|caption| value_1576
    style value_1576 fill:#f9f,stroke:#666
    value_1576["Loan Terms"]:::valueNode
    loanTermsTable -->|columnHeader| value_1576
    style value_1576 fill:#f9f,stroke:#666
    value_6914["Amount"]:::valueNode
    loanTermsTable -->|columnHeader| value_6914
    style value_6914 fill:#f9f,stroke:#666
    value_7034["Loan Amount"]:::valueNode
    loanAmount -->|rowHeader| value_7034
    style value_7034 fill:#f9f,stroke:#666
    value_1343["$680,000.00"]:::valueNode
    loanAmount -->|cellValue| value_1343
    style value_1343 fill:#f9f,stroke:#666
    value_1343["$680,000.00"]:::valueNode
    loanAmount -->|amount| value_1343
    style value_1343 fill:#f9f,stroke:#666
    value_1939["Interest Rate"]:::valueNode
    interestRate -->|rowHeader| value_1939
    style value_1939 fill:#f9f,stroke:#666
    value_2002["6.750%"]:::valueNode
    interestRate -->|cellValue| value_2002
    style value_2002 fill:#f9f,stroke:#666
    value_2564["Monthly Principal & Interest"]:::valueNode
    monthlyPayment -->|rowHeader| value_2564
    style value_2564 fill:#f9f,stroke:#666
    value_1300["$4,409.85"]:::valueNode
    monthlyPayment -->|cellValue| value_1300
    style value_1300 fill:#f9f,stroke:#666
    value_1300["$4,409.85"]:::valueNode
    monthlyPayment -->|amount| value_1300
    style value_1300 fill:#f9f,stroke:#666
    value_3157["Estimated Total Closing Costs"]:::valueNode
    closingCosts -->|rowHeader| value_3157
    style value_3157 fill:#f9f,stroke:#666
    value_4574["$18,750.00"]:::valueNode
    closingCosts -->|cellValue| value_4574
    style value_4574 fill:#f9f,stroke:#666
    value_4574["$18,750.00"]:::valueNode
    closingCosts -->|amount| value_4574
    style value_4574 fill:#f9f,stroke:#666
    value_6031["January 30, 2026"]:::valueNode
    closingCosts -->|value| value_6031
    style value_6031 fill:#f9f,stroke:#666
    value_6031["January 30, 2026"]:::valueNode
    closingCosts -->|Date| value_6031
    style value_6031 fill:#f9f,stroke:#666
    value_5297["LOAN-2026-00423"]:::valueNode
    closingCosts -->|value| value_5297
    style value_5297 fill:#f9f,stroke:#666
    value_5297["LOAN-2026-00423"]:::valueNode
    closingCosts -->|id| value_5297
    style value_5297 fill:#f9f,stroke:#666
    value_2082["SARAH JOHNSON"]:::valueNode
    closingCosts -->|value| value_2082
    style value_2082 fill:#f9f,stroke:#666
    value_2082["SARAH JOHNSON"]:::valueNode
    closingCosts -->|name| value_2082
    style value_2082 fill:#f9f,stroke:#666
    value_3659["ABC CORPORATION"]:::valueNode
    closingCosts -->|value| value_3659
    style value_3659 fill:#f9f,stroke:#666
    value_3659["ABC CORPORATION"]:::valueNode
    closingCosts -->|name| value_3659
    style value_3659 fill:#f9f,stroke:#666
    value_6725["CV-2026-12345"]:::valueNode
    closingCosts -->|value| value_6725
    style value_6725 fill:#f9f,stroke:#666
    value_6725["CV-2026-12345"]:::valueNode
    closingCosts -->|id| value_6725
    style value_6725 fill:#f9f,stroke:#666
    value_219["January 5, 2026"]:::valueNode
    closingCosts -->|value| value_219
    style value_219 fill:#f9f,stroke:#666
    value_219["January 5, 2026"]:::valueNode
    closingCosts -->|Date| value_219
    style value_219 fill:#f9f,stroke:#666
    value_3674["January 15, 2026"]:::valueNode
    closingCosts -->|value| value_3674
    style value_3674 fill:#f9f,stroke:#666
    value_3674["January 15, 2026"]:::valueNode
    closingCosts -->|Date| value_3674
    style value_3674 fill:#f9f,stroke:#666
    value_8946["TechVentures Inc."]:::valueNode
    closingCosts -->|value| value_8946
    style value_8946 fill:#f9f,stroke:#666
    value_8946["TechVentures Inc."]:::valueNode
    closingCosts -->|name| value_8946
    style value_8946 fill:#f9f,stroke:#666
    value_6970["Alexandra Martinez"]:::valueNode
    closingCosts -->|value| value_6970
    style value_6970 fill:#f9f,stroke:#666
    value_6970["Alexandra Martinez"]:::valueNode
    closingCosts -->|name| value_6970
    style value_6970 fill:#f9f,stroke:#666
    value_7457["$185,000.00"]:::valueNode
    closingCosts -->|value| value_7457
    style value_7457 fill:#f9f,stroke:#666
    value_7457["$185,000.00"]:::valueNode
    closingCosts -->|MonetaryAmount| value_7457
    style value_7457 fill:#f9f,stroke:#666
    value_7987["February 1, 2026"]:::valueNode
    closingCosts -->|value| value_7987
    style value_7987 fill:#f9f,stroke:#666
    value_7987["February 1, 2026"]:::valueNode
    closingCosts -->|Date| value_7987
    style value_7987 fill:#f9f,stroke:#666
    value_7310["4532-8765-0012"]:::valueNode
    closingCosts -->|value| value_7310
    style value_7310 fill:#f9f,stroke:#666
    value_7310["4532-8765-0012"]:::valueNode
    closingCosts -->|id| value_7310
    style value_7310 fill:#f9f,stroke:#666
    value_456["456 Market St, San Francisco, ..."]:::valueNode
    closingCosts -->|value| value_456
    style value_456 fill:#f9f,stroke:#666
    value_456["456 Market St, San Francisco, ..."]:::valueNode
    closingCosts -->|address| value_456
    style value_456 fill:#f9f,stroke:#666
    value_586["David Kim"]:::valueNode
    closingCosts -->|value| value_586
    style value_586 fill:#f9f,stroke:#666
    value_586["David Kim"]:::valueNode
    closingCosts -->|name| value_586
    style value_586 fill:#f9f,stroke:#666
    value_2126["January 25, 2026"]:::valueNode
    closingCosts -->|value| value_2126
    style value_2126 fill:#f9f,stroke:#666
    value_2126["January 25, 2026"]:::valueNode
    closingCosts -->|Date| value_2126
    style value_2126 fill:#f9f,stroke:#666
    value_5153["Usage Summary"]:::valueNode
    usageSummaryTable -->|caption| value_5153
    style value_5153 fill:#f9f,stroke:#666
    value_4614["Service"]:::valueNode
    usageSummaryTable -->|columnHeader| value_4614
    style value_4614 fill:#f9f,stroke:#666
    value_452["Usage"]:::valueNode
    usageSummaryTable -->|columnHeader| value_452
    style value_452 fill:#f9f,stroke:#666
    value_7740["Rate"]:::valueNode
    usageSummaryTable -->|columnHeader| value_7740
    style value_7740 fill:#f9f,stroke:#666
    value_4014["Charges"]:::valueNode
    usageSummaryTable -->|columnHeader| value_4014
    style value_4014 fill:#f9f,stroke:#666
    value_771["Electric Service"]:::valueNode
    electricService -->|cellValue| value_771
    style value_771 fill:#f9f,stroke:#666
    value_7171["650 kWh"]:::valueNode
    electricService -->|cellValue| value_7171
    style value_7171 fill:#f9f,stroke:#666
    value_5889["$0.18/kWh"]:::valueNode
    electricService -->|cellValue| value_5889
    style value_5889 fill:#f9f,stroke:#666
    value_5472["$117.00"]:::valueNode
    electricService -->|cellValue| value_5472
    style value_5472 fill:#f9f,stroke:#666
    value_5472["$117.00"]:::valueNode
    electricService -->|amount| value_5472
    style value_5472 fill:#f9f,stroke:#666
    value_6246["Gas Service"]:::valueNode
    gasService -->|cellValue| value_6246
    style value_6246 fill:#f9f,stroke:#666
    value_3773["45 therms"]:::valueNode
    gasService -->|cellValue| value_3773
    style value_3773 fill:#f9f,stroke:#666
    value_8956["$1.25/therm"]:::valueNode
    gasService -->|cellValue| value_8956
    style value_8956 fill:#f9f,stroke:#666
    value_2504["$56.25"]:::valueNode
    gasService -->|cellValue| value_2504
    style value_2504 fill:#f9f,stroke:#666
    value_2504["$56.25"]:::valueNode
    gasService -->|amount| value_2504
    style value_2504 fill:#f9f,stroke:#666
    value_5047["Service Fees"]:::valueNode
    serviceFees -->|cellValue| value_5047
    style value_5047 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    serviceFees -->|cellValue| value_0
    style value_0 fill:#f9f,stroke:#666
    value_3814["$14.20"]:::valueNode
    serviceFees -->|cellValue| value_3814
    style value_3814 fill:#f9f,stroke:#666
    value_3814["$14.20"]:::valueNode
    serviceFees -->|amount| value_3814
    style value_3814 fill:#f9f,stroke:#666
    value_9142["TOTAL AMOUNT DUE"]:::valueNode
    totalUtility -->|rowHeader| value_9142
    style value_9142 fill:#f9f,stroke:#666
    value_9809["$187.45"]:::valueNode
    totalUtility -->|cellValue| value_9809
    style value_9809 fill:#f9f,stroke:#666
    value_9809["$187.45"]:::valueNode
    totalUtility -->|amount| value_9809
    style value_9809 fill:#f9f,stroke:#666
    value_7987["February 1, 2026"]:::valueNode
    totalUtility -->|value| value_7987
    style value_7987 fill:#f9f,stroke:#666
    value_7987["February 1, 2026"]:::valueNode
    totalUtility -->|Date| value_7987
    style value_7987 fill:#f9f,stroke:#666
    value_7480["Transaction History"]:::valueNode
    creditCardTransactions -->|caption| value_7480
    style value_7480 fill:#f9f,stroke:#666
    value_9700["Date"]:::valueNode
    creditCardTransactions -->|columnHeader| value_9700
    style value_9700 fill:#f9f,stroke:#666
    value_5495["Merchant"]:::valueNode
    creditCardTransactions -->|columnHeader| value_5495
    style value_5495 fill:#f9f,stroke:#666
    value_6914["Amount"]:::valueNode
    creditCardTransactions -->|columnHeader| value_6914
    style value_6914 fill:#f9f,stroke:#666
    value_4670["12/06/25"]:::valueNode
    ccTrans1 -->|cellValue| value_4670
    style value_4670 fill:#f9f,stroke:#666
    value_4670["12/06/25"]:::valueNode
    ccTrans1 -->|date| value_4670
    style value_4670 fill:#f9f,stroke:#666
    value_5881["Amazon.com"]:::valueNode
    ccTrans1 -->|cellValue| value_5881
    style value_5881 fill:#f9f,stroke:#666
    value_5443["$156.78"]:::valueNode
    ccTrans1 -->|cellValue| value_5443
    style value_5443 fill:#f9f,stroke:#666
    value_5443["$156.78"]:::valueNode
    ccTrans1 -->|amount| value_5443
    style value_5443 fill:#f9f,stroke:#666
    value_2464["12/08/25"]:::valueNode
    ccTrans2 -->|cellValue| value_2464
    style value_2464 fill:#f9f,stroke:#666
    value_2464["12/08/25"]:::valueNode
    ccTrans2 -->|date| value_2464
    style value_2464 fill:#f9f,stroke:#666
    value_2218["Whole Foods Market"]:::valueNode
    ccTrans2 -->|cellValue| value_2218
    style value_2218 fill:#f9f,stroke:#666
    value_5284["$87.45"]:::valueNode
    ccTrans2 -->|cellValue| value_5284
    style value_5284 fill:#f9f,stroke:#666
    value_5284["$87.45"]:::valueNode
    ccTrans2 -->|amount| value_5284
    style value_5284 fill:#f9f,stroke:#666
    value_7209["12/10/25"]:::valueNode
    ccTrans3 -->|cellValue| value_7209
    style value_7209 fill:#f9f,stroke:#666
    value_7209["12/10/25"]:::valueNode
    ccTrans3 -->|date| value_7209
    style value_7209 fill:#f9f,stroke:#666
    value_7065["Shell Gas Station"]:::valueNode
    ccTrans3 -->|cellValue| value_7065
    style value_7065 fill:#f9f,stroke:#666
    value_2464["$62.50"]:::valueNode
    ccTrans3 -->|cellValue| value_2464
    style value_2464 fill:#f9f,stroke:#666
    value_2464["$62.50"]:::valueNode
    ccTrans3 -->|amount| value_2464
    style value_2464 fill:#f9f,stroke:#666
    value_5974["12/15/25"]:::valueNode
    ccTrans4 -->|cellValue| value_5974
    style value_5974 fill:#f9f,stroke:#666
    value_5974["12/15/25"]:::valueNode
    ccTrans4 -->|date| value_5974
    style value_5974 fill:#f9f,stroke:#666
    value_9000["Apple Store"]:::valueNode
    ccTrans4 -->|cellValue| value_9000
    style value_9000 fill:#f9f,stroke:#666
    value_2495["$1,299.00"]:::valueNode
    ccTrans4 -->|cellValue| value_2495
    style value_2495 fill:#f9f,stroke:#666
    value_2495["$1,299.00"]:::valueNode
    ccTrans4 -->|amount| value_2495
    style value_2495 fill:#f9f,stroke:#666
    value_914["12/20/25"]:::valueNode
    ccTrans5 -->|cellValue| value_914
    style value_914 fill:#f9f,stroke:#666
    value_914["12/20/25"]:::valueNode
    ccTrans5 -->|date| value_914
    style value_914 fill:#f9f,stroke:#666
    value_8883["Starbucks"]:::valueNode
    ccTrans5 -->|cellValue| value_8883
    style value_8883 fill:#f9f,stroke:#666
    value_3288["$8.75"]:::valueNode
    ccTrans5 -->|cellValue| value_3288
    style value_3288 fill:#f9f,stroke:#666
    value_3288["$8.75"]:::valueNode
    ccTrans5 -->|amount| value_3288
    style value_3288 fill:#f9f,stroke:#666
    value_2764["$15,000.00"]:::valueNode
    ccTrans5 -->|value| value_2764
    style value_2764 fill:#f9f,stroke:#666
    value_2764["$15,000.00"]:::valueNode
    ccTrans5 -->|MonetaryAmount| value_2764
    style value_2764 fill:#f9f,stroke:#666
    value_6784["$12,612.08"]:::valueNode
    ccTrans5 -->|value| value_6784
    style value_6784 fill:#f9f,stroke:#666
    value_6784["$12,612.08"]:::valueNode
    ccTrans5 -->|MonetaryAmount| value_6784
    style value_6784 fill:#f9f,stroke:#666
    value_8492["2026-001-SF"]:::valueNode
    ccTrans5 -->|value| value_8492
    style value_8492 fill:#f9f,stroke:#666
    value_8492["2026-001-SF"]:::valueNode
    ccTrans5 -->|id| value_8492
    style value_8492 fill:#f9f,stroke:#666
    value_2659["005"]:::valueNode
    ccTrans5 -->|value| value_2659
    style value_2659 fill:#f9f,stroke:#666
    value_2659["005"]:::valueNode
    ccTrans5 -->|id| value_2659
    style value_2659 fill:#f9f,stroke:#666
    value_3458["December 31, 2025"]:::valueNode
    ccTrans5 -->|value| value_3458
    style value_3458 fill:#f9f,stroke:#666
    value_3458["December 31, 2025"]:::valueNode
    ccTrans5 -->|Date| value_3458
    style value_3458 fill:#f9f,stroke:#666
    value_3763["Construction Payment Applicati..."]:::valueNode
    constructionPaymentApp -->|caption| value_3763
    style value_3763 fill:#f9f,stroke:#666
    value_8097["Item"]:::valueNode
    constructionPaymentApp -->|columnHeader| value_8097
    style value_8097 fill:#f9f,stroke:#666
    value_1427["Description"]:::valueNode
    constructionPaymentApp -->|columnHeader| value_1427
    style value_1427 fill:#f9f,stroke:#666
    value_9776["Scheduled Value"]:::valueNode
    constructionPaymentApp -->|columnHeader| value_9776
    style value_9776 fill:#f9f,stroke:#666
    value_6364["Work Completed"]:::valueNode
    constructionPaymentApp -->|columnHeader| value_6364
    style value_6364 fill:#f9f,stroke:#666
    value_4680["Balance"]:::valueNode
    constructionPaymentApp -->|columnHeader| value_4680
    style value_4680 fill:#f9f,stroke:#666
    value_7794["01 General Requirements"]:::valueNode
    constructionItem1 -->|cellValue| value_7794
    style value_7794 fill:#f9f,stroke:#666
    value_2819["Site mobilization, temp facili..."]:::valueNode
    constructionItem1 -->|cellValue| value_2819
    style value_2819 fill:#f9f,stroke:#666
    value_4734["$450,000.00"]:::valueNode
    constructionItem1 -->|cellValue| value_4734
    style value_4734 fill:#f9f,stroke:#666
    value_4734["$450,000.00"]:::valueNode
    constructionItem1 -->|amount| value_4734
    style value_4734 fill:#f9f,stroke:#666
    value_4734["$450,000.00"]:::valueNode
    constructionItem1 -->|cellValue| value_4734
    style value_4734 fill:#f9f,stroke:#666
    value_4734["$450,000.00"]:::valueNode
    constructionItem1 -->|amount| value_4734
    style value_4734 fill:#f9f,stroke:#666
    value_6865["$0.00"]:::valueNode
    constructionItem1 -->|cellValue| value_6865
    style value_6865 fill:#f9f,stroke:#666
    value_6865["$0.00"]:::valueNode
    constructionItem1 -->|amount| value_6865
    style value_6865 fill:#f9f,stroke:#666
    value_6534["03 Concrete"]:::valueNode
    constructionItem2 -->|cellValue| value_6534
    style value_6534 fill:#f9f,stroke:#666
    value_4641["Foundation and structural conc..."]:::valueNode
    constructionItem2 -->|cellValue| value_4641
    style value_4641 fill:#f9f,stroke:#666
    value_4408["$2,850,000.00"]:::valueNode
    constructionItem2 -->|cellValue| value_4408
    style value_4408 fill:#f9f,stroke:#666
    value_4408["$2,850,000.00"]:::valueNode
    constructionItem2 -->|amount| value_4408
    style value_4408 fill:#f9f,stroke:#666
    value_8022["$1,995,000.00"]:::valueNode
    constructionItem2 -->|cellValue| value_8022
    style value_8022 fill:#f9f,stroke:#666
    value_8022["$1,995,000.00"]:::valueNode
    constructionItem2 -->|amount| value_8022
    style value_8022 fill:#f9f,stroke:#666
    value_6315["$855,000.00"]:::valueNode
    constructionItem2 -->|cellValue| value_6315
    style value_6315 fill:#f9f,stroke:#666
    value_6315["$855,000.00"]:::valueNode
    constructionItem2 -->|amount| value_6315
    style value_6315 fill:#f9f,stroke:#666
    value_1522["05 Metals"]:::valueNode
    constructionItem3 -->|cellValue| value_1522
    style value_1522 fill:#f9f,stroke:#666
    value_8765["Structural steel and metal fra..."]:::valueNode
    constructionItem3 -->|cellValue| value_8765
    style value_8765 fill:#f9f,stroke:#666
    value_508["$3,200,000.00"]:::valueNode
    constructionItem3 -->|cellValue| value_508
    style value_508 fill:#f9f,stroke:#666
    value_508["$3,200,000.00"]:::valueNode
    constructionItem3 -->|amount| value_508
    style value_508 fill:#f9f,stroke:#666
    value_9322["$1,280,000.00"]:::valueNode
    constructionItem3 -->|cellValue| value_9322
    style value_9322 fill:#f9f,stroke:#666
    value_9322["$1,280,000.00"]:::valueNode
    constructionItem3 -->|amount| value_9322
    style value_9322 fill:#f9f,stroke:#666
    value_7579["$1,920,000.00"]:::valueNode
    constructionItem3 -->|cellValue| value_7579
    style value_7579 fill:#f9f,stroke:#666
    value_7579["$1,920,000.00"]:::valueNode
    constructionItem3 -->|amount| value_7579
    style value_7579 fill:#f9f,stroke:#666
    value_7649["TOTALS"]:::valueNode
    constructionTotals -->|rowHeader| value_7649
    style value_7649 fill:#f9f,stroke:#666
    value_9262["$8,500,000.00"]:::valueNode
    constructionTotals -->|cellValue| value_9262
    style value_9262 fill:#f9f,stroke:#666
    value_9262["$8,500,000.00"]:::valueNode
    constructionTotals -->|amount| value_9262
    style value_9262 fill:#f9f,stroke:#666
    value_3631["$3,725,000.00"]:::valueNode
    constructionTotals -->|cellValue| value_3631
    style value_3631 fill:#f9f,stroke:#666
    value_3631["$3,725,000.00"]:::valueNode
    constructionTotals -->|amount| value_3631
    style value_3631 fill:#f9f,stroke:#666
    value_6154["$4,775,000.00"]:::valueNode
    constructionTotals -->|cellValue| value_6154
    style value_6154 fill:#f9f,stroke:#666
    value_6154["$4,775,000.00"]:::valueNode
    constructionTotals -->|amount| value_6154
    style value_6154 fill:#f9f,stroke:#666
    value_6182["$372,500.00"]:::valueNode
    constructionTotals -->|value| value_6182
    style value_6182 fill:#f9f,stroke:#666
    value_6182["$372,500.00"]:::valueNode
    constructionTotals -->|MonetaryAmount| value_6182
    style value_6182 fill:#f9f,stroke:#666
    value_1677["$2,800,000.00"]:::valueNode
    constructionTotals -->|value| value_1677
    style value_1677 fill:#f9f,stroke:#666
    value_1677["$2,800,000.00"]:::valueNode
    constructionTotals -->|MonetaryAmount| value_1677
    style value_1677 fill:#f9f,stroke:#666
    value_6276["(512) 555-0199"]:::valueNode
    constructionTotals -->|value| value_6276
    style value_6276 fill:#f9f,stroke:#666
    value_6276["(512) 555-0199"]:::valueNode
    constructionTotals -->|phone| value_6276
    style value_6276 fill:#f9f,stroke:#666
    value_6144["(415) 555-0142"]:::valueNode
    constructionTotals -->|value| value_6144
    style value_6144 fill:#f9f,stroke:#666
    value_6144["(415) 555-0142"]:::valueNode
    constructionTotals -->|phone| value_6144
    style value_6144 fill:#f9f,stroke:#666
    value_219["January 5, 2026"]:::valueNode
    constructionTotals -->|value| value_219
    style value_219 fill:#f9f,stroke:#666
    value_219["January 5, 2026"]:::valueNode
    constructionTotals -->|Date| value_219
    style value_219 fill:#f9f,stroke:#666
    value_4959["VEND-8472"]:::valueNode
    constructionTotals -->|value| value_4959
    style value_4959 fill:#f9f,stroke:#666
    value_4959["VEND-8472"]:::valueNode
    constructionTotals -->|id| value_4959
    style value_4959 fill:#f9f,stroke:#666
    value_6222["Purchase Order Line Items"]:::valueNode
    purchaseOrderItems -->|caption| value_6222
    style value_6222 fill:#f9f,stroke:#666
    value_7504["Item #"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_7504
    style value_7504 fill:#f9f,stroke:#666
    value_7645["Part Number"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_7645
    style value_7645 fill:#f9f,stroke:#666
    value_1427["Description"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_1427
    style value_1427 fill:#f9f,stroke:#666
    value_3545["Qty"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_3545
    style value_3545 fill:#f9f,stroke:#666
    value_1215["Unit Price"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_1215
    style value_1215 fill:#f9f,stroke:#666
    value_3100["Ext. Price"]:::valueNode
    purchaseOrderItems -->|columnHeader| value_3100
    style value_3100 fill:#f9f,stroke:#666
    value_6505["1"]:::valueNode
    poItem1 -->|cellValue| value_6505
    style value_6505 fill:#f9f,stroke:#666
    value_8235["SKU-4587-A"]:::valueNode
    poItem1 -->|cellValue| value_8235
    style value_8235 fill:#f9f,stroke:#666
    value_2364["Heavy Duty Ball Bearing - 2" D..."]:::valueNode
    poItem1 -->|cellValue| value_2364
    style value_2364 fill:#f9f,stroke:#666
    value_9142["500"]:::valueNode
    poItem1 -->|cellValue| value_9142
    style value_9142 fill:#f9f,stroke:#666
    value_9140["$12.50"]:::valueNode
    poItem1 -->|cellValue| value_9140
    style value_9140 fill:#f9f,stroke:#666
    value_9140["$12.50"]:::valueNode
    poItem1 -->|amount| value_9140
    style value_9140 fill:#f9f,stroke:#666
    value_5460["$6,250.00"]:::valueNode
    poItem1 -->|cellValue| value_5460
    style value_5460 fill:#f9f,stroke:#666
    value_5460["$6,250.00"]:::valueNode
    poItem1 -->|amount| value_5460
    style value_5460 fill:#f9f,stroke:#666
    value_3572["2"]:::valueNode
    poItem2 -->|cellValue| value_3572
    style value_3572 fill:#f9f,stroke:#666
    value_5193["SKU-9821-B"]:::valueNode
    poItem2 -->|cellValue| value_5193
    style value_5193 fill:#f9f,stroke:#666
    value_6164["Stainless Steel Fastener Kit (..."]:::valueNode
    poItem2 -->|cellValue| value_6164
    style value_6164 fill:#f9f,stroke:#666
    value_8964["100"]:::valueNode
    poItem2 -->|cellValue| value_8964
    style value_8964 fill:#f9f,stroke:#666
    value_1564["$45.99"]:::valueNode
    poItem2 -->|cellValue| value_1564
    style value_1564 fill:#f9f,stroke:#666
    value_1564["$45.99"]:::valueNode
    poItem2 -->|amount| value_1564
    style value_1564 fill:#f9f,stroke:#666
    value_9353["$4,599.00"]:::valueNode
    poItem2 -->|cellValue| value_9353
    style value_9353 fill:#f9f,stroke:#666
    value_9353["$4,599.00"]:::valueNode
    poItem2 -->|amount| value_9353
    style value_9353 fill:#f9f,stroke:#666
    value_1695["3"]:::valueNode
    poItem3 -->|cellValue| value_1695
    style value_1695 fill:#f9f,stroke:#666
    value_5019["SKU-1456-C"]:::valueNode
    poItem3 -->|cellValue| value_5019
    style value_5019 fill:#f9f,stroke:#666
    value_8184["Industrial Lubricant - 5 Gallo..."]:::valueNode
    poItem3 -->|cellValue| value_8184
    style value_8184 fill:#f9f,stroke:#666
    value_2972["50"]:::valueNode
    poItem3 -->|cellValue| value_2972
    style value_2972 fill:#f9f,stroke:#666
    value_6152["$78.50"]:::valueNode
    poItem3 -->|cellValue| value_6152
    style value_6152 fill:#f9f,stroke:#666
    value_6152["$78.50"]:::valueNode
    poItem3 -->|amount| value_6152
    style value_6152 fill:#f9f,stroke:#666
    value_7600["$3,925.00"]:::valueNode
    poItem3 -->|cellValue| value_7600
    style value_7600 fill:#f9f,stroke:#666
    value_7600["$3,925.00"]:::valueNode
    poItem3 -->|amount| value_7600
    style value_7600 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    poSubtotal -->|rowHeader| value_0
    style value_0 fill:#f9f,stroke:#666
    value_5767["$14,774.00"]:::valueNode
    poSubtotal -->|cellValue| value_5767
    style value_5767 fill:#f9f,stroke:#666
    value_5767["$14,774.00"]:::valueNode
    poSubtotal -->|amount| value_5767
    style value_5767 fill:#f9f,stroke:#666
    value_0[""]:::valueNode
    poShipping -->|rowHeader| value_0
    style value_0 fill:#f9f,stroke:#666
    value_9373["$350.00"]:::valueNode
    poShipping -->|cellValue| value_9373
    style value_9373 fill:#f9f,stroke:#666
    value_9373["$350.00"]:::valueNode
    poShipping -->|amount| value_9373
    style value_9373 fill:#f9f,stroke:#666
    value_960["TOTAL:"]:::valueNode
    poTotal -->|rowHeader| value_960
    style value_960 fill:#f9f,stroke:#666
    value_6490["$15,124.00"]:::valueNode
    poTotal -->|cellValue| value_6490
    style value_6490 fill:#f9f,stroke:#666
    value_6490["$15,124.00"]:::valueNode
    poTotal -->|amount| value_6490
    style value_6490 fill:#f9f,stroke:#666
    value_4542["Thomas Chen, Procurement Manag..."]:::valueNode
    poTotal -->|value| value_4542
    style value_4542 fill:#f9f,stroke:#666
    value_4542["Thomas Chen, Procurement Manag..."]:::valueNode
    poTotal -->|name| value_4542
    style value_4542 fill:#f9f,stroke:#666
    value_9849["Deliver to Receiving Dock B. N..."]:::valueNode
    poTotal -->|value| value_9849
    style value_9849 fill:#f9f,stroke:#666
    value_1935["$7,400.00"]:::valueNode
    poTotal -->|value| value_1935
    style value_1935 fill:#f9f,stroke:#666
    value_1935["$7,400.00"]:::valueNode
    poTotal -->|MonetaryAmount| value_1935
    style value_1935 fill:#f9f,stroke:#666
    value_7762["(415) 555-0180"]:::valueNode
    poTotal -->|value| value_7762
    style value_7762 fill:#f9f,stroke:#666
    value_7762["(415) 555-0180"]:::valueNode
    poTotal -->|phone| value_7762
    style value_7762 fill:#f9f,stroke:#666
    value_8841["Amanda K."]:::valueNode
    poTotal -->|value| value_8841
    style value_8841 fill:#f9f,stroke:#666
    value_8841["Amanda K."]:::valueNode
    poTotal -->|name| value_8841
    style value_8841 fill:#f9f,stroke:#666
    value_2632["12"]:::valueNode
    poTotal -->|value| value_2632
    style value_2632 fill:#f9f,stroke:#666
    value_2495["01/05/26 7:45 PM"]:::valueNode
    poTotal -->|value| value_2495
    style value_2495 fill:#f9f,stroke:#666
    value_2495["01/05/26 7:45 PM"]:::valueNode
    poTotal -->|Date| value_2495
    style value_2495 fill:#f9f,stroke:#666
    value_7620["2x Caesar Salad"]:::valueNode
    poTotal -->|value| value_7620
    style value_7620 fill:#f9f,stroke:#666
    value_904["1x Grilled Salmon"]:::valueNode
    poTotal -->|value| value_904
    style value_904 fill:#f9f,stroke:#666
    value_2673["1x Filet Mignon"]:::valueNode
    poTotal -->|value| value_2673
    style value_2673 fill:#f9f,stroke:#666
    value_8361["2x House Wine"]:::valueNode
    poTotal -->|value| value_8361
    style value_8361 fill:#f9f,stroke:#666
    value_525["alexandra.martinez@email.com"]:::valueNode
    poTotal -->|value| value_525
    style value_525 fill:#f9f,stroke:#666
    value_525["alexandra.martinez@email.com"]:::valueNode
    poTotal -->|email| value_525
    style value_525 fill:#f9f,stroke:#666
    value_6144["(415) 555-0142"]:::valueNode
    poTotal -->|value| value_6144
    style value_6144 fill:#f9f,stroke:#666
    value_6144["(415) 555-0142"]:::valueNode
    poTotal -->|phone| value_6144
    style value_6144 fill:#f9f,stroke:#666
    value_8117["San Francisco, CA"]:::valueNode
    poTotal -->|value| value_8117
    style value_8117 fill:#f9f,stroke:#666
    value_8117["San Francisco, CA"]:::valueNode
    poTotal -->|address| value_8117
    style value_8117 fill:#f9f,stroke:#666
    value_8946["TechVentures Inc."]:::valueNode
    poTotal -->|value| value_8946
    style value_8946 fill:#f9f,stroke:#666
    value_8946["TechVentures Inc."]:::valueNode
    poTotal -->|name| value_8946
    style value_8946 fill:#f9f,stroke:#666
    value_1697["StartupXYZ"]:::valueNode
    poTotal -->|value| value_1697
    style value_1697 fill:#f9f,stroke:#666
    value_1697["StartupXYZ"]:::valueNode
    poTotal -->|name| value_1697
    style value_1697 fill:#f9f,stroke:#666
    value_7508["B.S. Computer Science"]:::valueNode
    poTotal -->|value| value_7508
    style value_7508 fill:#f9f,stroke:#666
    value_7563["Stanford University"]:::valueNode
    poTotal -->|value| value_7563
    style value_7563 fill:#f9f,stroke:#666
    value_7563["Stanford University"]:::valueNode
    poTotal -->|name| value_7563
    style value_7563 fill:#f9f,stroke:#666
    value_609["{{ version }}"]:::valueNode
    poTotal -->|value| value_609
    style value_609 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    poTotal -->|value| value_5357
    style value_5357 fill:#f9f,stroke:#666
    value_5357["{{ generation_date }}"]:::valueNode
    poTotal -->|Date| value_5357
    style value_5357 fill:#f9f,stroke:#666

    %% Hierarchy (containment)
    transactionsTable -.->|contains| transaction___loop_index___
    transactionsTable -.->|contains| employmentInfo
    employmentInfo -.->|contains| employeeName
    employmentInfo -.->|contains| jobTitle
    employmentInfo -.->|contains| company
    employmentInfo -.->|contains| salary
    employmentInfo -.->|contains| ssn
    employmentInfo -.->|contains| dob
    employmentInfo -.->|contains| employeeAddress
    signatureSection -.->|contains| attorneySignature
    notary -.->|contains| commissionNum
    notary -.->|contains| witness
    invoiceHeader -.->|contains| invoiceNumber
    invoiceHeader -.->|contains| invoiceDate
    invoiceHeader -.->|contains| dueDate
    invoiceItemsTable -.->|contains| item1
    item1 -.->|contains| item2
    item2 -.->|contains| subtotalRow
    subtotalRow -.->|contains| taxRow
    taxRow -.->|contains| totalRow
    earningsTable -.->|contains| regularPay
    regularPay -.->|contains| grossPay
    grossPay -.->|contains| deductionsTable
    deductionsTable -.->|contains| fedTax
    fedTax -.->|contains| stateTax
    stateTax -.->|contains| ssn
    ssn -.->|contains| medicare
    medicare -.->|contains| totalDeductions
    totalDeductions -.->|contains| netPay
    loanTermsTable -.->|contains| loanAmount
    loanAmount -.->|contains| interestRate
    interestRate -.->|contains| monthlyPayment
    monthlyPayment -.->|contains| closingCosts
    usageSummaryTable -.->|contains| electricService
    electricService -.->|contains| gasService
    gasService -.->|contains| serviceFees
    serviceFees -.->|contains| totalUtility
    creditCardTransactions -.->|contains| ccTrans1
    ccTrans1 -.->|contains| ccTrans2
    ccTrans2 -.->|contains| ccTrans3
    ccTrans3 -.->|contains| ccTrans4
    ccTrans4 -.->|contains| ccTrans5
    constructionPaymentApp -.->|contains| constructionItem1
    constructionItem1 -.->|contains| constructionItem2
    constructionItem2 -.->|contains| constructionItem3
    constructionItem3 -.->|contains| constructionTotals
    purchaseOrderItems -.->|contains| poItem1
    poItem1 -.->|contains| poItem2
    poItem2 -.->|contains| poItem3
    poItem3 -.->|contains| poSubtotal
    poSubtotal -.->|contains| poShipping
    poShipping -.->|contains| poTotal

    classDef typedNode fill:#fff,stroke:#333,stroke-width:2px
    classDef valueNode fill:#f9f,stroke:#666,stroke-dasharray: 5 5
```
