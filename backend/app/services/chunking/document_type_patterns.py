"""Extra section headings per classified document type (structure-based chunking)."""

TYPE_EXTRA_HEADINGS: dict[str, list[str]] = {
    "Employment Contract": [
        "Probation",
        "Non-compete",
        "Salary",
        "Benefits",
        "Intellectual Property",
        "Garden Leave",
    ],
    "Lease Agreement": [
        "Rent",
        "Security Deposit",
        "Premises",
        "Subletting",
        "Maintenance",
        "Lessor",
        "Lessee",
    ],
    "Rental Agreement": [
        "Tenant",
        "Landlord",
        "Rent",
        "Security Deposit",
        "Utilities",
    ],
    "NDA": [
        "Confidential Information",
        "Disclosure",
        "Return of Materials",
        "Injunctive Relief",
    ],
    "Vendor Agreement": [
        "Purchase Order",
        "Delivery",
        "Warranty",
        "Audit",
        "Supplier",
    ],
    "Service Agreement": [
        "Statement of Work",
        "SLA",
        "Service Level",
        "Acceptance",
        "Milestone",
    ],
    "General Legal Document": [],
}
