import base64
import calendar
import glob
import os
import shutil
import unicodedata

import altair as alt
import fitz
import pandas as pd
import plotly.express as px
import streamlit as st
from dateutil import parser
from PIL import Image
from PyPDF2 import PdfFileReader

# from SessionState import _get_state


data_fld = "data"
raw_fld = os.path.join(data_fld, "raw")
clean_fld = os.path.join(data_fld, "clean")
filter_selection = [
    "",
    "Date",
    "Price",
    "Vendor",
    "Category",
    "Account",
    "Comment",
    "File",
]


def calculator(expansion=st):

    container = expansion.container()

    calc_dict = {
        "Cash": [1000, 500, 200, 100, 50, 20],
        "Coins": [10, 5, 2, 1, 0.5, 0.05, 0.01],
    }

    total = 0

    for k, v in calc_dict.items():
        titles_with_amounts = container.empty()
        columns = [*container.columns(len(v))]
        t_amt = 0

        for i, col in enumerate(columns):
            base_amt = v[i]

            with col:
                x = st.text_input(label=f"{base_amt:,}")

                if x.isdigit():
                    calc = int(x) * base_amt
                    st.text(f"{calc:,}")
                    t_amt += calc
                    total += calc

        titles_with_amounts.header(f"{k}: {t_amt:,}")

    container.header(f"Total: {round(total, 2):,}")


@st.cache
def prep_files():  # sourcery no-metrics

    if os.path.exists(clean_fld):
        shutil.rmtree(clean_fld)
    shutil.copytree(raw_fld, clean_fld)

    # keyword lists
    months = [calendar.month_name[i] for i in range(1, 13)]
    month_index = {
        month: index for index, month in enumerate(calendar.month_name) if month
    }
    month_index_reverse = {
        index: month for index, month in enumerate(calendar.month_name) if month
    }

    acct_keywords = set()
    for f in os.listdir(clean_fld):
        name = f.split(".")[0]
        words = name.split()
        acct_keywords.update(words)
    acct_keywords = list(acct_keywords)

    for f in sorted(os.listdir(clean_fld)):
        if f.startswith("."):
            continue
        # prep
        split = f.split(".")
        name = split[0]
        extension = split[1]

        # strip trailing whitespace.
        name = name.strip()

        # iterate througn name words
        words = name.split()
        new_words = []

        previous = None

        for w in words:

            # deal with extra uncessary spaces
            if previous is None:
                # first element.
                pass
            elif previous + w in months + acct_keywords:
                # append separated words ie "Fe bruary"
                new_words.pop()
                w = previous + w

            # Capture Date, to append after loop
            if "-" in w:
                parts = w.split("-")
                if all(p.isdigit() for p in parts):
                    if len(parts) == 2:
                        parts.append("01")
                    date = parser.parse("-".join(parts), yearfirst=True)
                    month = month_index_reverse[date.month]
                    year = date.year
                    continue
            if w in months:
                month = w
                continue

            if w.isdigit() and len(w) == 4:
                year = w
                continue

            # Change Special Characters.
            try:
                w = unicode(w, "utf-8")
            except NameError:  # unicode is a default on python 3
                pass

            w = (
                unicodedata.normalize("NFD", w)
                .encode("ascii", "ignore")
                .decode("utf-8")
            )

            if w in ["Associac?a?o", "Associação", "Associacao"]:
                w = "AAM"

            if w == "Saving":
                w = "Savings"

            new_words.append(w)
            previous = w

        # Remove Words
        for exclude in ["Overland", "Enterprise", "Perimeter", "Bag", "PC"]:
            if exclude in new_words:
                new_words.remove(exclude)

        if "AAM" in new_words:
            new_words.remove("AAM")
            new_words.insert(0, "AAM")
        else:
            if "OE" in new_words:
                new_words.remove("OE")

            if "Kitty" in new_words:
                new_words.remove("Kitty")
                new_words.insert(0, "KITTY")
            elif "Expeditions" in new_words:
                new_words.insert(0, "AAM")
            else:
                new_words.insert(0, "OE")

        date_list = [str(year), str(month_index[month]).zfill(2)]
        new_name = " ".join(new_words + date_list)
        new_f = new_name + "." + extension
        old_filepath = os.path.join(clean_fld, f)
        new_filepath = os.path.join(clean_fld, new_f)
        os.rename(old_filepath, new_filepath)


def _group_names(dff):

    change = {
        "ALBERTO BENNET": "ALBERTO (BENNET)",
        "ALEXANDRE MANDANGO": "ALEXANDRE FRANCISCO MADANGO",
        "AMG": "AMG, LDA",
        "BUILDERS WAREHOUSE MAPUTO": "BUILDERS",
        "CATER KING": ["CATER KING LDA", "CATER KING, LDA"],
        "CLOSING BALANCE": [
            "CLOSE OUT JANUARY",
            "CLOSE OUT FEBRUARY",
            "CLOSE OUT MARCH",
            "CLOSE OUT APRIL",
            "CLOSE OUT MAY",
            "CLOSE OUT JUNE",
            "CLOSE OUT JULY",
            "CLOSE OUT AUGUST",
            "CLOSE OUT SEPTEMBER",
            "CLOSE OUT OCTOBER",
            "CLOSE OUT NOVEMBER",
            "CLOSE OUT DECEMBER",
            "CLOSE OUT MONTH",
        ],
        "COMERCIO AZUL": "COMERCIO AZUL LDA",
        "CONSERVATORIA": "MINISTERIO DA JUSTICA CONSERVATORIA",
        "EDM": "EDM EP",
        "J.M. TRADING": "J.M. TRADING,LDA",
        "JOSE COVELA": "JOSE NGUILA COVELA",
        "LIDIA BIA": "LIDIA FRANCISCO BIA",
        "MACHAVENGA": "MACHAVENGA LDA",
        "MAXIXE MART": "MAXIXE MART, LDA",
        "OVERLAND ENTERPRISES": "OVERLAND ENTERPRISES, LDA",
        "PADARIA BARBALARZA": "BARBALARZA PADARIA",
        "PEDRO JOSE": "PEDRO JOSE CUAMBA",
        "SOFEMA": ["SOFEMA LDA", "SOFEMA, LDA"],
        "SOMOCONTA": "SOMOCONTA LDA",
        "SUPERMERCADO NUMBER ONE": "SUPERMERCADO #1",
        "TX FROM BCI": "TO FROM BCI",
        "TX FROM AAM BASE": [
            "TX FROM ASSOCIACAO PC BASE",
            "TX FROM ASSOCIACAO PC",
        ],
        "TX FROM AAM BCI": "TX FROM ASSOCIACAO BCI",
        "TX FROM BASE": [
            "TX FROM BASE BAG",
            "TX FROM AAM EXPEDITIONS",
            "TX FROM OE BASE",
            "TX FROM BASE SAVINGS",
        ],
        "TX FROM AAM EXPEDITIONS": [
            "TX FROM EXPEDITIONS",
            "TX FROM EXPEDITIONS ASSOCIACAO",
        ],
        "TX FROM KITTY ROLLOVER": "TX FROM ROLLOVER",
        "TX TO AAM BCI": "TX TO ASSOC BCI",
        "TX TO AAM BASE": ["TX TO ASSOCIACAO BASE", "TX TO PC ASSOCIACAO"],
        "TX TO BASE": ["TX TO BASE BAG", "TX TO OVERLAND ENTERPRISE BASE"],
        "TX TO WALL": ["TX TO ENTERPRISE WALL", "TX TO PETTY CASH PROJECTS"],
        "TX TO AAM EXPEDITIONS": "TX TO EXPEDITIONS",
        "TX TO KITTY ROLLOVER": "TX TO ROLLOVER",
        "WOOLWORTHS MAPUTO": "WOOLWORTHS",
    }

    for k, v in change.items():
        dff["Name"] = dff["Name"].replace(v, k)

    return dff


def _normalize_col(dff, col):
    if col in dff.columns:
        dff[col] = (
            dff[col]
            .str.strip()
            .str.replace("\s{2,}", " ", regex=True)
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
            .str.upper()
        )
    else:
        st.write(f"{col} does not exist in df.")
    return dff


@st.cache
def get_df():
    # Create df of all CSV's to normalize them.
    all_files = glob.glob(os.path.join(clean_fld, "*.csv"))
    li = []
    for filename in all_files:
        dff = pd.read_csv(filename)
        fname = os.path.basename(filename)
        name = fname.split(".")[0]
        dff["filename"] = name
        dff["fileacct"] = " ".join(name.split()[:-2])
        dff["filedate"] = "-".join(name.split()[-2:])
        li.append(dff)

    df = pd.concat(li, axis=0, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["filedate"] = pd.to_datetime(df["filedate"]).dt.date

    df = df.drop(
        columns=[
            "Price",
            "Currency",
            "Exchange Rate",
            "Payment Method",
            "Category Code",
        ]
    ).rename(columns={"Price (Exchanged)": "Price"})
    df["Price"] = df["Price"].round(2)

    df = _normalize_col(df, "Name")
    df = _normalize_col(df, "Comment")
    df = _group_names(df)

    return df.sort_values(by=["filedate", "Receipt Index"]).reset_index(drop=True)


def _get_df_date_range(dff, agg="Months", range_type="Range"):

    "Date Range selector, filters main_df"

    if agg == "Days":
        date_col = "Date"
        date_format_func = lambda x: f"{x:%b %d, %Y}"
    elif agg == "Months":
        date_col = "filedate"
        date_format_func = lambda x: f"{x:%b %Y}"  # Format date example: "Dec 2020"

    dates = sorted(dff[date_col].unique())
    date_min = dff[date_col].min()
    date_max = dff[date_col].max()

    if range_type == "Range":
        date_range = st.select_slider(
            "Date Range",
            dates,
            value=(date_min, date_max),
            format_func=date_format_func,
        )
        date_filter = dff[date_col].between(*date_range)

    elif range_type == "Single":
        date_select = st.select_slider(
            "Date Single",
            dates,
            value=date_max,
            format_func=date_format_func,
        )
        date_filter = dff[date_col].eq(date_select)

    return dff[date_filter]


def filter_selected(col, current):
    # Preferable
    # All options remain for filters, but if a used filter is selected,
    # first removes previous position.
    f = col.selectbox(
        "",
        filter_selection,
        index=filter_selection.index(current),
    )

    current_index = st.session_state.filters_selected.index(current)
    if f == "":
        st.session_state.filters_selected.pop(current_index)
        st.experimental_rerun()

    elif f != current:
        st.session_state.filters_selected.pop(current_index)
        while f in st.session_state.filters_selected:
            st.session_state.filters_selected.remove(f)
        st.session_state.filters_selected.insert(current_index, f)
        st.experimental_rerun()


def filter_selected2(col, current):
    # Less preferable
    # Removes used options from filters.
    # Must use "" option to relieve availability to use again.
    remove_list = st.session_state.filters_selected.copy()
    remove_list.remove(current)
    filter_selection_remove_used = [x for x in filter_selection if x not in remove_list]
    f = col.selectbox(
        "",
        filter_selection_remove_used,
        index=filter_selection_remove_used.index(current),
    )

    current_index = st.session_state.filters_selected.index(current)
    if f == "":
        st.session_state.filters_selected.pop(current_index)
        st.experimental_rerun()
    elif f != current:
        st.session_state.filters_selected.pop(current_index)
        st.session_state.filters_selected.insert(current_index, f)
        st.experimental_rerun()


def date_filter(dff, expansion=st):

    container = expansion.container()

    # Date selectboxes and slider
    f, date_agg_col, range_or_single_col, slider_col = container.columns((1, 1, 1, 5))
    filter_selected(f, "Date")

    date_agg = date_agg_col.selectbox("Dates", options=["Days", "Months"])
    range_or_single = range_or_single_col.selectbox(
        "Range or Single", options=["Range", "Single"]
    )

    with slider_col:
        df = _get_df_date_range(dff, agg=date_agg, range_type=range_or_single)

    return df, date_agg, range_or_single


def date_filter_on_graph(dff, expansion=st):
    container = expansion.container()

    # Date selectboxes and slider
    date_agg_col, range_or_single_col, slider_col = container.columns((1, 1, 6))

    date_agg = date_agg_col.selectbox("Dates", options=["Days", "Months"])
    range_or_single = range_or_single_col.selectbox(
        "Range or Single", options=["Range", "Single"]
    )

    with slider_col:
        df = _get_df_date_range(dff, agg=date_agg, range_type=range_or_single)

    return df, date_agg, range_or_single


def _is_digit(x):
    # can validate negative digits unlike builtin x.isdigit()
    try:
        int(x)
        return True
    except ValueError:
        return False


def price_filter(dff, expansion=st):

    container = expansion.container()

    f, p_min, p_max, price_slider = container.columns((1, 1, 1, 5))
    filter_selected(f, "Price")

    price_min = int(round(dff["Price"].min(), 0) - 1)
    price_max = int(round(dff["Price"].max(), 0) + 1)
    p_min_amt = p_min.text_input("Price Min", value=f"{price_min:,}")
    p_max_amt = p_max.text_input("Price Max", value=f"{price_max:,}")

    value_min = int(p_min_amt) if _is_digit(p_min_amt) else price_min
    value_max = int(p_max_amt) if _is_digit(p_max_amt) else price_max

    # slider cant format numbers with comma, but select_slider converts all to strings.
    price_range = price_slider.slider(
        "Price Range",
        min_value=value_min,
        max_value=value_max,
        value=(value_min, value_max),
        step=1,
        format="%d",  # %d %e %f %g %i
    )

    price_f = dff["Price"].between(*price_range)

    dff = dff[price_f]

    return dff


def comment_filter(dff, expansion=st):

    container = expansion.container()

    f, comments = container.columns((1, 7))
    filter_selected(f, "Comment")

    comments_string = comments.text_input("Comment Search")

    if comments_string:
        comments_f = dff["Comment"].str.contains(comments_string, case=False, na=False)
        dff = dff[comments_f]
    return dff


def multiselect_filters(dff, filter_name, col_name, expansion=st):
    container = expansion.container()

    f, filter_chosen = container.columns((1, 7))
    filter_selected(f, filter_name)

    filter_options = sorted(dff[col_name].unique())
    filters_selected = filter_chosen.multiselect(
        label=f"{col_name} ({len(filter_options)})", options=filter_options
    )
    if filters_selected:
        dff = dff[dff[col_name].isin(filters_selected)]
    return dff


def filters(df, expansion=st):

    container = expansion.container()

    if "filters_selected" not in st.session_state:
        st.session_state.filters_selected = []

    date_agg = "Days"
    range_or_single = "Range"

    mutliselect_filters = {
        "Vendor": "Name",
        "Category": "Category Name",
        "File": "filename",
        "Account": "fileacct",
    }

    for filter_chosen in st.session_state.filters_selected:

        if filter_chosen == "Comment":
            df = comment_filter(df, expansion=container)

        elif filter_chosen == "Date":
            df, date_agg, range_or_single = date_filter(df, expansion=container)

        elif filter_chosen == "Price":
            df = price_filter(df, expansion=container)

        elif filter_chosen in mutliselect_filters:
            df = multiselect_filters(
                df,
                filter_name=filter_chosen,
                col_name=mutliselect_filters[filter_chosen],
                expansion=container,
            )

    add_filter, leave_blank = container.columns([1, 7])

    filters_remove_selected = [
        x for x in filter_selection if x not in st.session_state.filters_selected
    ]

    filter_add = add_filter.selectbox(
        "Add Filter",
        filters_remove_selected,
        index=filters_remove_selected.index(""),
    )

    if filter_add != "":
        st.session_state.filters_selected.append(filter_add)
        st.experimental_rerun()

    return df, date_agg, range_or_single


def columns_to_show(dff, expansion=st):

    container = expansion.container()

    label = "Show Columns" if expansion == st else ""

    show_columns, reset_columns = container.columns((7, 1))

    cols = list(dff.columns)

    if "filter_cols" not in st.session_state:
        st.session_state.filter_cols = 0

    reset_columns.markdown("")  # Spacing
    reset_columns.markdown("")  # Spacing
    if reset_columns.button("Reset"):
        st.session_state.filter_cols += 1

    return show_columns.multiselect(
        label,
        options=cols,
        default=cols,
        key=str(st.session_state.filter_cols),
    )


def see_receipts(dff):
    receipt_indexs = dff.index
    receipt_id = st.sidebar.selectbox("See Receipt", receipt_indexs)
    if receipt_id:
        anchor_link("linkto_receipt", "Link to Receipt")

    return receipt_id


@st.cache
def get_df_running_totals(dff):
    # Graph df needs some rows removed.
    # st.title("Graph_df")
    # Keep first Beginning Balance in earliest csv, and remove all the consecutive ones.

    # dff["Name"] = dff["Name"].str.strip()
    dff.loc[:, "Name"] = dff["Name"].str.strip().str.normalize("NFKD")
    # dff["Name"] = dff["Name"].str.normalize("NFKD")  # unicode cleaning
    # dff.loc[:, "Name"] = dff["Name"].str.normalize("NFKD")

    graph_df = dff[
        ~(dff.duplicated(["Name", "fileacct"]) & dff["Name"].eq("BEGINNING BALANCE"))
    ]

    # If bb date is not the earliest, then remove bb row
    # idx = graph_df.groupby(["fileacct", "filedate"])["filedate"].transform(min) ==
    # for a in picked:
    #     dff = graph_df[graph_df["fileacct"] == a]
    #     st.write(a)
    #     st.dataframe(dff.groupby(["filedate"]).min())

    # Remove all closing balance rows.
    # get max receipt index based on filename
    # check if BF - Dep in Categorical Code
    # and "Close" or Closing in Name
    idx = (
        graph_df.groupby(["filename"])["Receipt Index"].transform(max)
        == graph_df["Receipt Index"]
    )
    graph_df_remove = graph_df[idx]
    graph_df_remove = graph_df_remove[
        graph_df_remove["Category Name"] == "BF - Deposit"
    ]
    graph_df_remove = graph_df_remove[
        graph_df_remove["Name"].str.contains("clos", case=False)
    ]
    graph_df = graph_df.drop(index=graph_df_remove.index)

    # Running sum of accounts
    graph_df["Running Total"] = graph_df["Price"].cumsum()

    return graph_df


def chart(dff):
    fig = px.line(dff, x="Date", y="Running Total", title="Account Summary")
    st.plotly_chart(fig)


def chart_altair(dff):
    return (
        alt.Chart(dff)
        .mark_line()
        .encode(
            alt.X("filedate:T"),
            y="Running Total",
        )
    )


def chart_altair_hist(dff, agg="Months", range_type="Range"):
    if agg == "Days":
        date_col = "Date"
        date_x = f"{date_col}:T"
        tooltip = [date_x]
    elif agg == "Months":
        date_col = "filedate"
        date_x = f"yearmonth({date_col}):T"
        tooltip = [alt.Tooltip(date_x, title="Month")]

    stacked = "fileacct"
    tooltip.append(alt.Tooltip(stacked, title="Account"))
    if len(dff["fileacct"].unique()) == 1:
        stacked = "Name"
        tooltip.append(stacked)
        if len(dff["Name"].unique()) == 1:
            # st.radio("Legend Options:", ["Category Name", "Comment"])
            # if len(dff["Comment"].unique()) >= 30:
            stacked = "Category Name"
            tooltip.append(alt.Tooltip("index", title="Receipt ID"))
            tooltip.append(alt.Tooltip("Date", title="Receipt Date"))
            tooltip.append("Category Name")
            tooltip.append("Comment")

    if range_type == "Range":
        x = alt.X(date_x)
        y = alt.Y("sum(Price):Q")
        interval_encoding = ["x"]
    elif range_type == "Single":
        x = alt.X("sum(Price):Q")
        y = alt.Y(f"{stacked}:O")
        interval_encoding = ["y"]

    legend_selection = alt.selection_multi(
        fields=[stacked], on="mouseover", bind="legend"
    )

    interval = alt.selection_interval(
        encodings=interval_encoding,
        bind="scales",
        on="[mousedown, mouseup] > mousemove",
    )

    return (
        alt.Chart(dff.reset_index())
        .mark_bar()
        .encode(
            x=x,
            y=y,
            color=alt.Color(f"{stacked}:O"),
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.2)),
            tooltip=[*tooltip, "sum(Price)"],
        )
        .add_selection(legend_selection, interval)
        .properties(height=500)
    )


def chart_altair_final(dff, agg):

    if agg == "Days":
        date_col = "Date"
    elif agg == "Months":
        date_col = "filedate"

    base = alt.Chart(dff).encode(x=alt.X(f"{date_col}:T"))

    bar = base.mark_bar().encode(
        y=alt.Y("sum(Price):Q"),
        color=alt.Color("fileacct:O"),
        tooltip=[date_col, "Name", "fileacct", "sum(Price)"],
    )

    line = base.mark_line(color="red").encode(y=alt.Y("Running Total:Q"))

    return (bar + line).interactive()


def split_df_into_accounts(dff, accts_picked):
    for a in accts_picked:
        split_df = dff[dff["fileacct"] == a]
        split_df["Running Total"] = split_df["Price"].cumsum()
        try:
            rt = split_df["Running Total"].iloc[-1]
        except:
            rt = 0
        header = f"RT: {a} ({rt:,.2f})"
        st.header(header)

        # box = f"checkbox_{'_'.join(a.split())}.text("
        # exec(box + f" = {rt})")

        st.dataframe(dff)


def display_pdf(pdf_file):

    """Displays pdf_file on streamlit. Error prone for any doc over 2mb"""

    # clean_fld = os.path.join("data", "clean")
    # pdf_file = os.path.join(clean_fld, pdf)

    if os.path.exists(pdf_file):
        # base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        with open(pdf_file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        # st.write("loaded")
    else:
        st.write(f"{pdf_file} does not exist.")
        return

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="1300" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)


def _create_pdf_dict(filename):

    doc = fitz.open(filename)
    all_d = {}
    for page in doc:
        num = page.number
        txt = page.get_text("text")
        txt = txt.split("\n")
        receipts = []
        for t in txt:
            if " • " in t and t not in receipts:
                receipts.append(t)
        if receipts:
            d = {}
            for r in receipts:  # Important to sort
                items = r.split(" • ")
                # st.write(r)
                d[int(items[0])] = {
                    "Vendor": items[1],
                    "Date": items[2],
                    "Comment": items[3],
                    "Page": num,
                }
            d_list = sorted(list(d.keys()))

            images = page.get_images()
            img_dict = {int(i[7][2:]): i[0] for i in images}
            img_dict_sorted = {i: img_dict[i] for i in sorted(img_dict)}

            # Append to receipt dict (d) the image info.
            for i, img in enumerate(img_dict_sorted):
                rec_id = d_list[i]
                d[rec_id]["img_name"] = img
                d[rec_id]["img_xref"] = img_dict_sorted[img]

            all_d = {**all_d, **d}

    return all_d


def _generate_pic(filename, receipt):

    xref = receipt["img_xref"]
    num = receipt["Page"]
    pic_name = receipt["img_name"]

    doc = fitz.open(filename)

    pix = fitz.Pixmap(doc, xref)
    if pix.n >= 5:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    pic_fld = os.path.join("data", "img")
    if os.path.exists(pic_fld):
        shutil.rmtree(pic_fld)
    os.mkdir(pic_fld)

    filename = f"page{num}_{pic_name}_xref{xref}.png"
    filepath = os.path.join(pic_fld, filename)

    pix.writePNG(filepath)

    return filepath


def _get_pic(filepath):

    return Image.open(filepath)


def _get_receipt_pic(pdf, receipt_index):

    clean_fld = os.path.join("data", "clean")
    filename = os.path.join(clean_fld, pdf)

    d = _create_pdf_dict(filename)

    try:
        receipt = d[receipt_index]
    except:
        return None

    pic_filepath = _generate_pic(filename, receipt)

    return _get_pic(pic_filepath)


def view_receipts(dff, picked_df, expansion=st):

    container = expansion.container()

    main_left, main_middle, main_right = container.columns([1, 1, 9])

    all_or_picked = main_left.radio("Pool From", ["All", "Picked"])

    chosen_df = dff if all_or_picked == "All" else picked_df
    index_list = chosen_df.index.to_list()

    if "receipt_position" not in st.session_state:
        st.session_state.receipt_position = 0

    receipt_id = main_middle.selectbox(
        "", index_list, index=st.session_state.receipt_position
    )
    st.session_state.receipt_position = index_list.index(receipt_id)

    # Previous Button Behavior
    if (st.session_state.receipt_position - 1) >= 0:
        if main_left.button("Previous"):
            st.session_state.receipt_position -= 1
            st.experimental_rerun()

    # Next Button Behavior
    if (st.session_state.receipt_position + 1) < len(index_list):
        if main_middle.button("Next"):
            st.session_state.receipt_position += 1
            st.experimental_rerun()

    main_right.table(dff.iloc[[receipt_id]])

    row = dff.iloc[receipt_id]
    filename = row["filename"]
    receipt_index = row["Receipt Index"]

    image = _get_receipt_pic(filename + ".pdf", receipt_index)
    if image:
        container.image(image, use_column_width=True)
    else:
        container.write("Receipt Does Not Exist")


def display_pdf2(pdf, receipt_index):

    """Displays Document by breaking up into pages, then reading each page as one doc."""

    tmp_fld = os.path.join("data", "tmp")
    if os.path.exists(tmp_fld):
        shutil.rmtree(tmp_fld)
    os.mkdir(tmp_fld)

    clean_fld = os.path.join("data", "clean")
    pdf_file = os.path.join(clean_fld, pdf)

    _get_receipt_pic(pdf_file, receipt_index)
    st.write("done")

    inputpdf = PdfFileReader(open(pdf_file, "rb"))

    for i in range(inputpdf.numPages):
        page = inputpdf.getPage(i)
        st.write(i)
        # st.write(page.extractText())
        text = page.extractText()
        text = text.split("\n")[1:-1]

        for t in text:
            t = t.split(" ¥ ")
            # print(t)
            # print(repr(t).split("¥"))

        print("\n")

        # output = PdfFileWriter()
        # output.addPage(inputpdf.getPage(i))
        # tmp_file = os.path.join(tmp_fld, "document-page%s.pdf" % i)

        # with open(tmp_file, "wb") as tmp:
        #     output.write(tmp)

        # display_pdf(tmp.name)


def anchor_link(name, description):
    st.sidebar.markdown(f"<a href='#{name}'>{description}</a>", unsafe_allow_html=True)


def anchor_point(name):
    st.markdown(f"<div id='{name}'></div>", unsafe_allow_html=True)


def main():

    s = "Calculator"
    ex_calculator = st.expander(s)
    calculator(expansion=ex_calculator)

    prep_files()  # Cache
    df_main = get_df()  # Cache

    s = "Filters"
    ex_filters = st.expander(s, expanded=True)
    df, date_agg, range_or_single = filters(
        df_main,
        expansion=ex_filters,
    )

    # Graph Date and Fig placeholder
    s = "Graph"
    ex_graph = st.expander(s, expanded=True)
    if "Date" not in st.session_state.filters_selected:
        df, date_agg, range_or_single = date_filter_on_graph(
            df,
            expansion=ex_graph,
        )

    graph_df = get_df_running_totals(df)  # TODO

    s = "Columns"
    ex_columns = st.expander(s)
    show_cols = columns_to_show(
        df_main,
        expansion=ex_columns,
    )

    # Accounts found in main df
    accounts = sorted(df["fileacct"].unique())
    picked = []
    for i in accounts:
        acct_rt = graph_df[graph_df["fileacct"] == i]["Price"].sum()
        if st.sidebar.checkbox(f"{i} ({acct_rt:,.2f})"):
            picked.append(i)

    picked_df = df[df["fileacct"].isin(picked)].sort_values(
        by=["Date", "Receipt Index"]
    )
    if picked:
        st.title("Picked")
        st.dataframe(picked_df.style.format({"Price": "{:.2f}"}))

    try:
        rt = graph_df["Running Total"].iloc[-1]
    except:
        rt = 0

    header = f"RT: All ({rt:,.2f})"
    st.header(header)
    st.dataframe(graph_df[show_cols].style.format({"Price": "{:.2f}"}))

    # filtered for graph
    picked_df = graph_df[graph_df["fileacct"].isin(picked)] if picked else graph_df

    fig = chart_altair_hist(picked_df, agg=date_agg, range_type=range_or_single)
    ex_graph.altair_chart(fig, use_container_width=True)

    s = "See Receipts"
    ex_receipts = st.expander(s)
    view_receipts(
        df_main,
        picked_df,
        expansion=ex_receipts,
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Mozi Finances", layout="wide", initial_sidebar_state="collapsed"
    )

    st.title("Mozi Finances")

    main()
