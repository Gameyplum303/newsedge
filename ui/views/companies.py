"""Companies view: each company's history status and editable relationship profile."""
import pandas as pd
import streamlit as st

from engine import companies, jobs, profiles, store
from .. import data
from ..state import Page

RELATIONS = list(companies.RELATION_SIGN)


def _status(page: Page):
    rows = []
    for t in page.controls.tickers:
        job = store.open_job(t)
        coverage = jobs.history_coverage(t)
        rows.append({
            "Ticker": t,
            "Company": ", ".join(page.controls.watchlist.get(t, [])),
            "History": "downloading" if job else ("ready" if coverage >= jobs.READY_COVERAGE else "incomplete"),
            "10-year news coverage": coverage * 100,
            "Articles": int((page.analysis.articles["ticker"] == t).sum()),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "10-year news coverage": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    })


def _editor(page: Page, ticker: str):
    p = profiles.profile(ticker)
    if p["industry"]:
        st.caption(f"{p['sector']} · {p['industry']}")

    aliases = st.text_input("Product and brand names (comma-separated)", ", ".join(p["aliases"]),
                            help="Names that identify the company in news that doesn't name it, e.g. AWS for Amazon. "
                                 "Avoid everyday words.", key=f"aliases_{ticker}")

    st.markdown("**Government business** (0 = none, 1 = most of the business)")
    cols = st.columns(len(companies.GOVERNMENT_AREA_LABELS))
    government = {area: col.slider(label[0].upper() + label[1:], 0.0, 1.0, float(p["government"].get(area, 0.0)), 0.05,
                                   key=f"gov_{area}_{ticker}")
                  for col, (area, label) in zip(cols, companies.GOVERNMENT_AREA_LABELS.items())}

    st.markdown("**Related companies**: news about them flows to this stock (competitors' news counts in reverse).")
    table = pd.DataFrame([{"Names": ", ".join(r["names"]), "Relation": r["relation"], "Weight": float(r["weight"])}
                          for r in p["relationships"]], columns=["Names", "Relation", "Weight"])
    edited = st.data_editor(table, num_rows="dynamic", hide_index=True, width="stretch", key=f"rel_{ticker}",
                            column_config={
                                "Names": st.column_config.TextColumn(help="Comma-separated, e.g. TSMC, Taiwan Semiconductor"),
                                "Relation": st.column_config.SelectboxColumn(options=RELATIONS, required=True),
                                "Weight": st.column_config.NumberColumn(min_value=0.05, max_value=1.0, step=0.05,
                                                                        help="How strongly their news affects this stock"),
                            })

    c1, c2 = st.columns(2)
    if c1.button("💾 Save profile", key=f"save_{ticker}", width="stretch"):
        relationships = [{"names": [n.strip() for n in str(r["Names"]).split(",") if n.strip()],
                          "relation": r["Relation"] or "partner", "weight": float(r["Weight"] or 0.3)}
                         for r in edited.to_dict("records") if str(r.get("Names") or "").strip()]
        profiles.save_profile(ticker, aliases=[a.strip() for a in aliases.split(",") if a.strip()],
                              government={k: v for k, v in government.items() if v > 0}, relationships=relationships)
        data.load_articles.clear()   # links are re-derived from the profile; the model retrains
        st.toast(f"{ticker} profile saved. Articles are re-linked and the model retrains.", icon="💾")
        st.rerun()
    if c2.button("🔁 Re-download related-company history", key=f"related_{ticker}", width="stretch",
                 help="After adding partners or government business, search 10 years of their news again."):
        store.reset_windows(ticker, "related")
        jobs.enqueue(ticker, page.controls.watchlist.get(ticker, []), kinds=["related"])
        st.toast(f"Searching {ticker}'s related-company history in the background.", icon="🔁")


def render(page: Page):
    st.markdown("Every stock you add downloads its 10-year news history in the background, and the model retrains "
                "when it's done. Review each company's profile below: it decides which partner, government and "
                "world news counts for the stock.")
    _status(page)
    st.subheader("Company profile")
    ticker = st.selectbox("Company", page.controls.tickers, key="profile_ticker")
    _editor(page, ticker)
    st.caption("Government business for newly added companies is suggested from their industry and description. "
               "Relationships can't be looked up reliably from free data, so add suppliers, customers, partners and "
               "competitors here.")
