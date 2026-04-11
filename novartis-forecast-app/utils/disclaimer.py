import streamlit as st


def show_synthetic_data_note():
    """Anonymization banner for pages with fully synthetic data."""
    st.warning(
        "**Transparency note** — All data on this page is fully synthetic. "
        "It was generated to match the statistical properties of the real dataset "
        "(same product count, franchise distribution, seasonality shape, and "
        "order-of-magnitude volumes). No real figures or identifiers are used. "
        "Approximated KPIs are prefixed with ~.",
        icon="⚠️",
    )


def show_study_note():
    """Anonymization banner for pages referencing the real study KPIs."""
    st.warning(
        "**Transparency note** — This study was conducted at a major French "
        "pharmaceutical company whose name has been anonymized for confidentiality. "
        "All KPIs shown have been slightly modified (rounded/approximated) to protect "
        "proprietary data. The magnitude and direction of all findings are fully preserved. "
        "Approximated values are prefixed with ~.",
        icon="⚠️",
    )
