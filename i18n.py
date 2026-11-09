"""Language of the coin spreadsheet (SUMMARY_LANGUAGE=EN|DE in .env, default EN).

Internal keys stay English everywhere (value_source, needs_review reasons, grades);
only the text written into the workbook is translated, through tr() and localize().
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()
LANG = os.environ.get("SUMMARY_LANGUAGE", "EN").strip().upper() or "EN"


def set_language(lang: str) -> None:
    global LANG
    LANG = lang.strip().upper()


def is_de() -> bool:
    return LANG == "DE"


# ------------------------------------------------------------- sheet / UI text

DE: Dict[str, str] = {
    # sheet names
    "Summary": "Übersicht",
    "Highlights": "Highlights",
    "Coins": "Münzen",
    "Types": "Typen",
    "Top": "Top",
    "Review": "Prüfen",
    # column headers
    "Picture": "Bild",
    "Coin ID": "Münz-ID",
    "Country": "Land",
    "Denomination": "Nominal",
    "Year": "Jahr",
    "Subject": "Motiv",
    "Est. value €": "Geschätzter Wert €",
    "Value source": "Wertquelle",
    "Melt €": "Metallwert €",
    "Numista € @grade": "Numista € (Erhaltung)",
    "Grade": "Erh.",
    "LLM guess €": "KI-Schätzung €",
    "Override €": "Manuell €",
    "F €": "s (F) €",
    "VF €": "ss (VF) €",
    "XF €": "vz (XF) €",
    "UNC €": "st (UNC) €",
    "Composition": "Material",
    "Weight g": "Gewicht g",
    "Silver g": "Silber g",
    "Gold g": "Gold g",
    "Numista type": "Numista-Typ",
    "Numista": "Numista",
    "eBay sold": "eBay verkauft",
    "KM#": "KM#",
    "Mintage": "Auflage",
    "Currency": "Währung",
    "Face value": "Nennwert",
    "Mint": "Prägest.",
    "Type": "Art",
    "Grade (guess)": "Erhaltung (geschätzt)",
    "LLM conf.": "KI-Sicherheit",
    "Match": "Zuordnung",
    "Needs review": "Zu prüfen",
    "Cluster": "Gruppe",
    "Front": "Vorders.",
    "Back": "Rücks.",
    "Front scan": "Scan vorne",
    "Back scan": "Scan hinten",
    "Numista ID": "Numista-ID",
    "Overridden": "Manuell geändert",
    "Notes": "Notizen",
    "Count": "Anzahl",
    "Value each €": "Wert je Stück €",
    "Total €": "Summe €",
    "Silver g total": "Silber g gesamt",
    "Coin IDs": "Münz-IDs",
    "open": "öffnen",
    # summary sheet
    "Valuation inputs": "Bewertungsgrundlagen",
    "fixed values at these prices": "feste Werte zu diesen Preisen",
    "edit B3/B4: melt and values recalculate": "B3/B4 ändern: Metall- und Gesamtwerte rechnen neu",
    "Prices taken from .env on": "Preise aus .env übernommen am",
    "Silver € per g (pure)": "Silber € pro g (fein)",
    "Gold € per g (pure)": "Gold € pro g (fein)",
    "Assumed grade (Numista price)": "Angenommene Erhaltung (Numista-Preis)",
    "(change in .env + re-run)": "(in .env ändern + neu erstellen)",
    "Numista prices fetched": "Numista-Preise abgerufen",
    "Collection": "Sammlung",
    "Identified by LLM": "Von KI bestimmt",
    "Matched to Numista": "Numista zugeordnet",
    "Distinct types": "Verschiedene Typen",
    "Need review": "Zu prüfen",
    "Highlights (confidently valuable)": "Highlights (sicher wertvoll)",
    "Total estimated value €": "Geschätzter Gesamtwert €",
    "  of which from LLM guesses €": "  davon aus KI-Schätzungen €",
    "Total melt value €": "Metallwert gesamt €",
    "Total silver g": "Silber gesamt g",
    "Total gold g": "Gold gesamt g",
    "How to read this": "Lesehilfe",
    "grade_legend": "s = schön (F) · ss = sehr schön (VF) · vz = vorzüglich (XF) · "
    "st = stempelglanz (UNC)",
    "source_legend": "Wertquelle: Numista = Katalogpreis · Metallwert = Edelmetallgehalt "
    "· KI-Schätzung = grobe Schätzung, nicht geprüft",
}


def tr(text: str) -> str:
    """UI text in the workbook language (English text is the key)."""
    return DE.get(text, text) if is_de() else text


def sheet_ref(name: str) -> str:
    """Sheet name for use in a formula: 'Münzen'!A1."""
    return f"'{tr(name)}'"


# ------------------------------------------------------------------ cell values

GRADE_DE = {
    "g": "ge",
    "vg": "sge",
    "f": "s",
    "vf": "ss",
    "xf": "vz",
    "au": "vz-st",
    "unc": "st",
}

SOURCE_DE = {
    "override": "manuell",
    "melt": "Metallwert",
    "llm_guess": "KI-Schätzung",
}


def grade_label(grade: str | None) -> str | None:
    if grade is None or not is_de():
        return grade
    return GRADE_DE.get(grade, grade)


def source_label(source: str | None) -> str | None:
    """'numista_xf' -> 'Numista vz', 'melt' -> 'Metallwert' (DE)."""
    if source is None or not is_de():
        return source
    if source.startswith("numista_"):
        return "Numista " + grade_label(source.removeprefix("numista_"))
    return SOURCE_DE.get(source, source)


REASONS_DE = {
    "not identified": "nicht bestimmt",
    "not a coin / excluded": "keine Münze / ausgeschlossen",
    "low confidence": "KI unsicher",
    "no Numista match": "kein Numista-Treffer",
    "ambiguous Numista match": "Numista-Treffer mehrdeutig",
    "cluster members disagree": "Gruppe uneinheitlich bestimmt",
    "metal differs from Numista type": "Metall weicht vom Numista-Typ ab",
    "Numista price ≫ LLM estimate: check issue/variant": "Numista-Preis ≫ KI-Schätzung: "
    "Jahrgang/Variante prüfen",
}


def reasons_label(text: str | None) -> str | None:
    if not text or not is_de():
        return text
    out = []
    for reason in text.split("; "):
        m = re.fullmatch(r"worth ≥ (\d+) €: verify", reason)
        out.append(
            f"Wert ≥ {m.group(1)} €: prüfen" if m else REASONS_DE.get(reason, reason)
        )
    return "; ".join(out)


ENUMS_DE = {
    # grade_guess from the LLM
    "circulated": "umlaufend",
    "extremely_fine": "vorzüglich",
    "uncirculated": "stempelglanz",
    "poor": "schlecht",
    "proof": "Polierte Platte",
    "unknown": "unbekannt",
    # coin_type
    "circulation": "Umlaufmünze",
    "commemorative": "Gedenkmünze",
    "token_or_medal": "Marke/Medaille",
    "bullion": "Anlagemünze",
    # match_status
    "matched": "zugeordnet",
    "matched_no_issue": "zugeordnet (ohne Jahrgang)",
    "no_candidates": "kein Treffer",
    "type_not_found": "Typ nicht gefunden",
    "pending_quota": "ausstehend",
    "excluded": "ausgeschlossen",
}

COUNTRY_DE = {
    "Albania": "Albanien",
    "Algeria": "Algerien",
    "Argentina": "Argentinien",
    "Australia": "Australien",
    "Austria": "Österreich",
    "Austria-Hungary": "Österreich-Ungarn",
    "Bavaria": "Bayern",
    "Belgium": "Belgien",
    "Bohemia and Moravia": "Böhmen und Mähren",
    "Bolivia": "Bolivien",
    "Brazil": "Brasilien",
    "British Caribbean Territories": "Britische Karibikgebiete",
    "British India": "Britisch-Indien",
    "British West Africa": "Britisch-Westafrika",
    "Bulgaria": "Bulgarien",
    "Burma": "Birma",
    "Canada": "Kanada",
    "Ceylon": "Ceylon",
    "Chile": "Chile",
    "China": "China",
    "Colombia": "Kolumbien",
    "Cyprus": "Zypern",
    "Czechoslovakia": "Tschechoslowakei",
    "Denmark": "Dänemark",
    "East Africa": "Ostafrika",
    "Ecuador": "Ecuador",
    "Egypt": "Ägypten",
    "Ethiopia": "Äthiopien",
    "Falkland Islands": "Falklandinseln",
    "Finland": "Finnland",
    "France": "Frankreich",
    "French West Africa": "Französisch-Westafrika",
    "Gambia": "Gambia",
    "Germany": "Deutschland",
    "Ghana": "Ghana",
    "Greece": "Griechenland",
    "Guernsey": "Guernsey",
    "Hanover": "Hannover",
    "Hong Kong": "Hongkong",
    "Hungary": "Ungarn",
    "Iceland": "Island",
    "India": "Indien",
    "Indonesia": "Indonesien",
    "Iran": "Iran",
    "Ireland": "Irland",
    "Isle of Man": "Isle of Man",
    "Israel": "Israel",
    "Italy": "Italien",
    "Japan": "Japan",
    "Jersey": "Jersey",
    "Jordan": "Jordanien",
    "Kenya": "Kenia",
    "Kingdom of Serbs, Croats and Slovenes": "Königreich der Serben, Kroaten und Slowenen",
    "Kuwait": "Kuwait",
    "Laos": "Laos",
    "Libya": "Libyen",
    "Lombardy-Venetia": "Lombardo-Venetien",
    "Luxembourg": "Luxemburg",
    "Malaysia": "Malaysia",
    "Malta": "Malta",
    "Mexico": "Mexiko",
    "Monaco": "Monaco",
    "Mongolia": "Mongolei",
    "Morocco": "Marokko",
    "Nepal": "Nepal",
    "Netherlands": "Niederlande",
    "Netherlands Antilles": "Niederländische Antillen",
    "New Zealand": "Neuseeland",
    "Norway": "Norwegen",
    "Ottoman Empire": "Osmanisches Reich",
    "Pakistan": "Pakistan",
    "Panama": "Panama",
    "Peru": "Peru",
    "Philippines": "Philippinen",
    "Poland": "Polen",
    "Portugal": "Portugal",
    "Prussia": "Preußen",
    "Rhodesia": "Rhodesien",
    "Romania": "Rumänien",
    "Russian Empire": "Russisches Kaiserreich",
    "Russian SFSR": "Russische SFSR",
    "Saarland": "Saarland",
    "Saint Thomas and Prince": "São Tomé und Príncipe",
    "Saxony": "Sachsen",
    "Singapore": "Singapur",
    "Slovakia": "Slowakei",
    "South Africa": "Südafrika",
    "South Korea": "Südkorea",
    "South Vietnam": "Südvietnam",
    "Soviet Union": "Sowjetunion",
    "Spain": "Spanien",
    "Sudan": "Sudan",
    "Sweden": "Schweden",
    "Switzerland": "Schweiz",
    "Syria": "Syrien",
    "Taiwan": "Taiwan",
    "Thailand": "Thailand",
    "Trinidad and Tobago": "Trinidad und Tobago",
    "Tunisia": "Tunesien",
    "Turkey": "Türkei",
    "Uganda": "Uganda",
    "United Arab Republic": "Vereinigte Arabische Republik",
    "United Kingdom": "Vereinigtes Königreich",
    "United States": "Vereinigte Staaten",
    "Uruguay": "Uruguay",
    "Vatican City": "Vatikanstadt",
    "Venezuela": "Venezuela",
    "West African States": "Westafrikanische Staaten",
    "Württemberg": "Württemberg",
    "Yugoslavia": "Jugoslawien",
    "Zeeland": "Zeeland",
    # qualifiers in brackets / after a dash
    "Empire": "Kaiserreich",
    "German Empire": "Deutsches Kaiserreich",
    "Third Reich": "Drittes Reich",
    "Weimar Republic": "Weimarer Republik",
    "Federal Republic": "Bundesrepublik",
    "Democratic Republic": "DDR",
    "People's Republic": "Volksrepublik",
    "Republic": "Republik",
    "Second Republic": "Zweite Republik",
    "Kingdom": "Königreich",
    "Holy Roman Empire": "Heiliges Römisches Reich",
    "Qing Dynasty": "Qing-Dynastie",
    "Vichy State": "Vichy-Regime",
    "German occupation": "deutsche Besatzung",
    "French Protectorate": "französisches Protektorat",
    "Portuguese colony": "portugiesische Kolonie",
    "British": "britisch",
    "Austrian": "österreichisch",
    "Mannheim": "Mannheim",
    "Notgeld": "Notgeld",
    "Baden": "Baden",
    "Hamburg": "Hamburg",
}


def country_label(country: str | None) -> str | None:
    """'Germany (Empire) - Prussia' -> 'Deutschland (Kaiserreich) - Preußen' (DE)."""
    if not country or not is_de():
        return country
    if country in COUNTRY_DE:
        return COUNTRY_DE[country]
    # translate the pieces between brackets and dashes, keep the punctuation
    parts = re.split(r"(\s*[()]\s*|\s*[-–]\s+)", country)
    return "".join(COUNTRY_DE.get(p, p) if p.strip() else p for p in parts)


def localize(key: str, value: Any) -> Any:
    """Cell value of column `key` in the workbook language."""
    if value is None or not is_de() or not isinstance(value, str):
        return value
    if key == "value_source":
        return source_label(value)
    if key == "price_grade":
        return grade_label(value)
    if key == "needs_review":
        return reasons_label(value)
    if key == "country":
        return country_label(value)
    if key in ("grade_guess", "coin_type", "match_status"):
        return ENUMS_DE.get(value, value)
    return value
