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
    "Open": "Offen",
    # column headers
    "Open: why": "Offen: Grund",
    "Next step": "Nächster Schritt",
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
    # selling plan (selling.py)
    "Selling plan": "Verkaufsplan",
    "Overview": "Übersicht",
    "Singles": "Einzelverkauf",
    "Precious metal": "Edelmetall",
    "Lots": "Lose",
    "Built": "Erstellt",
    "Route": "Verkaufsweg",
    "Catalog value €": "Katalogwert €",
    "Expected €": "Erwarteter Erlös €",
    "Sales": "Verkäufe",
    "Total": "Gesamt",
    "Sell individually": "Einzeln verkaufen",
    "Gold: precious-metal dealer": "Gold: Edelmetallhändler",
    "Silver: precious-metal dealer by weight": "Silber: Edelmetallhändler nach Gewicht",
    "Themed lots (base metal)": "Themen-Lose (unedle Metalle)",
    "Bulk / by the kilo": "Masse / nach Kilo",
    "Where on the page": "Lage auf der Seite",
    "Page": "Seite",
    "Row": "Reihe",
    "Pos.": "Pos.",
    "Coin": "Münze",
    "Value €": "Wert €",
    "Extra vs bulk €": "Mehrerlös ggü. Masse €",
    "Sell via": "Verkauf über",
    "Check before selling": "Vor dem Verkauf prüfen",
    "Metal": "Metall",
    "Fineness": "Feingehalt",
    "Lot": "Los",
    "silver": "Silber",
    "gold": "Gold",
    "weight unknown": "Gewicht unbekannt",
    "World mix": "Welt-Mix",
    "dealer / auction": "Händler / Auktion",
    "value is an LLM guess": "Wert nur KI-Schätzung",
    "Numista match ambiguous": "Numista-Treffer mehrdeutig",
    "duplicates identified differently": "Dubletten unterschiedlich bestimmt",
    "metal differs": "Metall weicht ab",
    "price ≫ LLM estimate": "Preis ≫ KI-Schätzung",
    "low LLM confidence": "KI unsicher",
    "see outlier_review.xlsx": "siehe outlier_review.xlsx",
    "location_help": "Fundort: Seite = Scan-Name ohne _v/_r, Reihe von oben, Position von links "
    "(Vorderseiten-Scan). Das Blatt Einzelverkauf zeigt die Seite mit markierter Münze.",
    "bulk_help": "Alle Münzen, die auf keinem Blatt stehen, gehen in die Masse (nach Kilo).",
    "details_help": "Schritte, Silber nach Feingehalt und Einstellungen: selling_guide.md",
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
    "Open: wait for Numista quota": "Offen: wartet auf Numista-Kontingent",
    "Open: identify manually": "Offen: manuell bestimmen",
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
    # open coins worksheet (open_coins.py)
    "Open coins": "Offene Münzen",
    "How to": "Anleitung",
    "Estimated coin": "Geschätzte Bezeichnung",
    "Estimated value €": "Geschätzter Wert €",
    "Numista suggestion": "Numista-Vorschlag",
    "My observations": "Meine Beobachtungen",
    "Status": "Stand",
    "sent to Claude, waiting for the result": "an Claude gesendet, Ergebnis ausstehend",
    "identified again": "neu bestimmt",
    "Numista type set by hand": "Numista-Typ von Hand gesetzt",
    "new note: will be sent with --submit": "neue Notiz: wird mit --submit gesendet",
    "help_title": "Offene Münzen: so geht's",
    "help_what": "Hier stehen die Münzen, die weder die KI noch der Numista-Katalog sicher "
    "bestimmen konnten, die wertvollsten zuerst. Fundort: Seite = Scan-Name ohne _v/_r, Reihe "
    "von oben, Position von links. Schau dir die Münze an und schreib in die gelbe Spalte "
    "„Meine Beobachtungen“, was du siehst. Deutsch ist in Ordnung.",
    "help_examples": "Hilfreich sind: Umschrift (auch nur Teile), Jahreszahl, Münzzeichen "
    "(Buchstabe wie A, D, F, G, J), Wertangabe, Material (magnetisch? Silber?), Gewicht in g, "
    "Durchmesser in mm, Rand (glatt, geriffelt, Schrift). Auch „ist eine Medaille“, "
    "„Kopie/Fälschung“ oder „stark abgegriffen“ hilft.",
    "help_numista": "Wenn du die Münze selbst auf numista.com gefunden hast, füg den Link ein "
    "(…/catalogue/pieces12345.html) oder schreib „N# 12345“. Dann wird genau dieser Typ "
    "verwendet.",
    "help_next": "Danach: Datei speichern und schließen. uv run open_coins.py --list zeigt "
    "kostenlos, welche Münzen gesendet würden und was es kostet; uv run open_coins.py --submit "
    "sendet sie an Claude, --collect --wait holt die Ergebnisse. Danach numista.py, "
    "build_table.py und open_coins.py neu ausführen.",
    "help_only": "Gesendet werden nur Münzen, bei denen du etwas eingetragen oder geändert "
    "hast. Deine Notizen werden zusätzlich in coin_notes.csv gesichert und bleiben beim "
    "Neuerstellen der Datei erhalten.",
}


EN = {
    "help_title": "Open coins: how this works",
    "help_what": "These coins could not be identified reliably by the LLM or the Numista "
    "catalogue, most valuable first. Where: page = scan name without _v/_r, row from the top, "
    "position from the left. Look at the coin and write what you see into the yellow column "
    "'My observations'. German is fine.",
    "help_examples": "Useful: legend (even parts of it), date, mint mark (letter such as A, D, "
    "F, G, J), denomination, metal (magnetic? silver?), weight in g, diameter in mm, edge "
    "(plain, reeded, lettered). 'It's a medal', 'copy/fake' or 'very worn' help too.",
    "help_numista": "If you found the coin on numista.com yourself, paste the link "
    "(…/catalogue/pieces12345.html) or write 'N# 12345'. Exactly that type is then used.",
    "help_next": "Then save and close the file. uv run open_coins.py --list shows for free "
    "which coins would be sent and what it costs; uv run open_coins.py --submit sends them to "
    "Claude, --collect --wait fetches the results. Afterwards re-run numista.py, "
    "build_table.py and open_coins.py.",
    "help_only": "Only coins whose note you added or changed are sent. Your notes are also "
    "saved to coin_notes.csv and survive rebuilding the file.",
    "location_help": "Where to find a coin: page = scan name without _v/_r, row from the top, "
    "position from the left (front scan). The Singles sheet shows the page with the coin marked.",
    "bulk_help": "Every coin that is on none of the sheets goes to bulk (sold by the kilo).",
    "details_help": "Steps, silver by fineness and settings: selling_guide.md",
}


def tr(text: str) -> str:
    """UI text in the workbook language (English text is the key)."""
    return DE.get(text, text) if is_de() else EN.get(text, text)


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
    "skipped_low_value": "übersprungen (geringer Wert)",
    "api_error": "API-Fehler",
    # open_reason / open_action (build_table.open_status)
    "not identified by LLM": "von KI nicht bestimmt",
    "not looked up on Numista yet": "noch nicht bei Numista abgefragt",
    "manual Numista type not fetched yet": "manueller Numista-Typ noch nicht abgerufen",
    "not found on Numista": "bei Numista nicht gefunden",
    "Numista match uncertain": "Numista-Treffer unsicher",
    "Numista has no price": "Numista hat keinen Preis",
    "wait for Numista quota": "warten (Numista-Kontingent)",
    "identify manually": "manuell bestimmen",
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
    "Russia / USSR": "Russland / UdSSR",
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
    if key in (
        "grade_guess",
        "coin_type",
        "match_status",
        "open_reason",
        "open_action",
    ):
        return ENUMS_DE.get(value, value)
    return value
