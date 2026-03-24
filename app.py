from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    session
)
import sqlite3
import os
import re
from datetime import datetime
from werkzeug.utils import secure_filename
import easyocr
import cv2

app = Flask(__name__)
app.secret_key = "expense-tracker-secret-key"

DATABASE = "database.db"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

reader = easyocr.Reader(["en", "tr"], gpu=False)


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_receipt_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return filepath

    h, w = image.shape[:2]
    max_width = 900
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "processed_" + os.path.basename(filepath)
    )
    cv2.imwrite(processed_path, gray)
    return processed_path


def extract_store_name(text_lines):
    known_stores = ["şok", "sok", "migros", "a101", "bim", "carrefour"]

    for line in text_lines[:8]:
        lower = line.lower()
        for store in known_stores:
            if store in lower:
                return store.upper()

    for line in text_lines[:5]:
        cleaned = line.strip()
        if len(cleaned) > 3 and not re.search(r"\d", cleaned):
            return cleaned.title()

    return "Unknown Store"


def normalize_number(text):
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def extract_total(text_lines):
    lines = [line.strip() for line in text_lines if line.strip()]

    def clean_token(token):
        return token.replace("*", "").replace("+", "").replace("'", "").strip()

    def to_float(x):
        try:
            return float(x.replace(",", "."))
        except ValueError:
            return None

    cleaned = [clean_token(x) for x in lines]

    toplam_idx = -1
    for i, token in enumerate(cleaned):
        low = token.lower()
        if "topkdv" in low:
            continue
        if "toplam" in low:
            toplam_idx = i
            break

    if toplam_idx != -1:
        window = []
        for j in range(toplam_idx + 1, min(toplam_idx + 15, len(cleaned))):
            low = cleaned[j].lower()
            if "nakit" in low or "para üstü" in low or "paraustu" in low:
                break
            window.append(cleaned[j])

        normal_numbers = []
        for token in window:
            m = re.search(r"(\d+[.,]\d{2})", token)
            if m:
                val = to_float(m.group(1))
                if val is not None and 1 <= val <= 100000:
                    normal_numbers.append(val)

        for val in normal_numbers:
            if val >= 100:
                return val

        whole_candidates = []
        local_decimals = []

        for token in cleaned:
            if re.fullmatch(r"\d+[.,]", token):
                whole_candidates.append(token)

        for token in window:
            if re.fullmatch(r"\d{2}", token):
                local_decimals.append(token)

        if whole_candidates and local_decimals:
            for wc in whole_candidates:
                whole = wc[:-1]
                if whole.isdigit():
                    whole_num = int(whole)
                    if 50 <= whole_num <= 100000:
                        for dec in reversed(local_decimals):
                            try:
                                val = float(f"{whole}.{dec}")
                                if 1 <= val <= 100000:
                                    return val
                            except ValueError:
                                pass

        if normal_numbers:
            best_small = normal_numbers[-1]
            if best_small < 100 and whole_candidates:
                for wc in whole_candidates:
                    whole = wc[:-1]
                    if whole.isdigit():
                        whole_num = int(whole)
                        if whole_num >= 100:
                            dec = str(best_small).split(".")[-1]
                            if len(dec) == 1:
                                dec = dec + "0"
                            return float(f"{whole_num}.{dec}")

        if normal_numbers:
            return normal_numbers[-1]

    for token in cleaned:
        m = re.search(r"(\d+[.,]\d{2})", token)
        if m:
            val = to_float(m.group(1))
            if val is not None and val >= 1:
                return val

    return 0.0


def extract_date(text_lines):
    for line in text_lines:
        match = re.search(r"(\d{2}[./-]\d{2}[./-]\d{4})", line)
        if match:
            raw = match.group(1)
            for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    parsed = datetime.strptime(raw, fmt)
                    return parsed.strftime("%Y-%m-%d")
                except ValueError:
                    pass
    return datetime.today().strftime("%Y-%m-%d")


def suggest_category(store_name):
    store = store_name.lower()

    if any(x in store for x in ["şok", "sok", "a101", "bim", "migros", "carrefour", "market"]):
        return "Grocery"
    if any(x in store for x in ["cafe", "kahve", "coffee", "burger", "pizza", "restaurant"]):
        return "Food"
    return "Other"


def ai_predict_category(store_name, note, transaction_type):
    text = f"{store_name} {note}".lower()

    if transaction_type == "income":
        if any(word in text for word in ["allowance", "harçlık", "harclik", "burs"]):
            return "Allowance"
        if any(word in text for word in ["salary", "maaş", "maas"]):
            return "Salary"
        if any(word in text for word in ["freelance", "project", "proje"]):
            return "Freelance"
        if any(word in text for word in ["gift", "hediye"]):
            return "Gift"
        if any(word in text for word in ["refund", "geri ödeme", "geri odeme"]):
            return "Refund"
        return "Other Income"

    if any(word in text for word in ["migros", "şok", "sok", "a101", "bim", "market", "pasar", "sebze", "meyve"]):
        return "Grocery"
    if any(word in text for word in ["cafe", "kahve", "coffee", "burger", "pizza", "restaurant", "makan", "yemek"]):
        return "Food"
    if any(word in text for word in ["taxi", "taksi", "metro", "otobüs", "otobus", "bus", "dolmuş", "dolmus", "parkir", "park"]):
        return "Transport"
    if any(word in text for word in ["shopping", "alışveriş", "alisveris", "giyim", "clothes"]):
        return "Shopping"
    if any(word in text for word in ["eczane", "hospital", "doktor", "health", "medicine", "ilaç", "ilac"]):
        return "Health"
    if any(word in text for word in ["kitap", "kurs", "okul", "education", "school"]):
        return "Education"

    return "Other"


def generate_ai_insights():
    conn = get_db_connection()

    total_expense = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='expense'"
    ).fetchone()[0]

    total_income = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='income'"
    ).fetchone()[0]

    top_category = conn.execute("""
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE type='expense'
        GROUP BY category
        ORDER BY total DESC
        LIMIT 1
    """).fetchone()

    conn.close()

    insights = []

    if total_income > total_expense and (total_income > 0 or total_expense > 0):
        insights.append("💰 Good job! Your income is higher than your expenses.")
    elif total_expense > total_income:
        insights.append("⚠️ Warning: Your expenses are higher than your income.")

    if top_category:
        category = top_category["category"]
        insights.append(f"📊 You spend the most on {category}.")

    if total_income > 0:
        saving_rate = (total_income - total_expense) / total_income
        if saving_rate > 0.3:
            insights.append("🔥 Excellent saving habit!")
        elif saving_rate < 0:
            insights.append("🚨 You are overspending!")
        else:
            insights.append("🙂 Your spending is moderate.")

    if total_expense == 0 and total_income == 0:
        insights.append("📭 No financial data yet. Start adding transactions!")

    return insights


def generate_monthly_ai_insights():
    conn = get_db_connection()

    monthly_data = conn.execute("""
        SELECT substr(date, 1, 7) as month,
               SUM(CASE WHEN type='income' THEN amount ELSE 0 END) as income,
               SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) as expense
        FROM transactions
        GROUP BY substr(date, 1, 7)
        ORDER BY month DESC
        LIMIT 2
    """).fetchall()

    conn.close()

    insights = []

    if len(monthly_data) >= 1:
        latest = monthly_data[0]
        latest_month = latest["month"]
        latest_income = float(latest["income"] or 0)
        latest_expense = float(latest["expense"] or 0)

        insights.append(
            f"📅 In {latest_month}, your income was {latest_income:.2f} and your expense was {latest_expense:.2f}."
        )

        if latest_income > latest_expense:
            insights.append("✅ This month you are still in a positive balance.")
        elif latest_expense > latest_income:
            insights.append("⚠️ This month your expenses are higher than your income.")

    if len(monthly_data) >= 2:
        latest = monthly_data[0]
        previous = monthly_data[1]

        latest_expense = float(latest["expense"] or 0)
        previous_expense = float(previous["expense"] or 0)

        if latest_expense > previous_expense:
            insights.append("📈 Your expenses increased compared to last month.")
        elif latest_expense < previous_expense:
            insights.append("📉 Your expenses decreased compared to last month.")
        else:
            insights.append("➖ Your expenses are similar to last month.")

    if not insights:
        insights.append("📭 Not enough monthly data yet.")

    return insights


def generate_chat_advice(user_message):
    conn = get_db_connection()

    total_income = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='income'"
    ).fetchone()[0]

    total_expense = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='expense'"
    ).fetchone()[0]

    top_category = conn.execute("""
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE type='expense'
        GROUP BY category
        ORDER BY total DESC
        LIMIT 1
    """).fetchone()

    monthly_data = conn.execute("""
        SELECT substr(date, 1, 7) as month,
               SUM(CASE WHEN type='income' THEN amount ELSE 0 END) as income,
               SUM(CASE WHEN type='expense' THEN amount ELSE 0 END) as expense
        FROM transactions
        GROUP BY substr(date, 1, 7)
        ORDER BY month DESC
        LIMIT 2
    """).fetchall()

    conn.close()

    text = user_message.lower()

    if "biggest" in text or "largest" in text or "en çok" in text or "biggest expense" in text:
        if top_category:
            return f"Your biggest expense category is {top_category['category']}, with a total spending of {float(top_category['total']):.2f}."
        return "I couldn't find enough expense data yet."

    if "budget" in text or "tips" in text or "öneri" in text or "tavsiy" in text:
        advice = []

        if total_income > 0:
            recommended_budget = total_income * 0.7
            advice.append(f"Try keeping your monthly spending below {recommended_budget:.2f}.")

        if top_category:
            advice.append(f"You currently spend the most on {top_category['category']}, so reducing that category may help you save more.")

        if total_expense > total_income:
            advice.append("Your expenses are higher than your income, so you should reduce non-essential spending.")
        else:
            advice.append("Your financial balance looks relatively stable. Keep tracking consistently.")

        return "Here are some personalized budgeting tips: " + " ".join(advice)

    if "income" in text and "expense" in text:
        balance = float(total_income) - float(total_expense)
        if balance >= 0:
            return f"Your total income is {float(total_income):.2f}, your total expense is {float(total_expense):.2f}, and your balance is {balance:.2f}. This means you are still in a positive financial position."
        return f"Your total income is {float(total_income):.2f}, your total expense is {float(total_expense):.2f}, and your balance is {balance:.2f}. This means your spending is currently higher than your income."

    if "month" in text or "bulan" in text or "ay" in text:
        if monthly_data:
            latest = monthly_data[0]
            latest_month = latest["month"]
            latest_income = float(latest["income"] or 0)
            latest_expense = float(latest["expense"] or 0)

            response = f"In {latest_month}, your income was {latest_income:.2f} and your expense was {latest_expense:.2f}. "

            if latest_income > latest_expense:
                response += "You are still in a positive balance this month."
            elif latest_expense > latest_income:
                response += "Your expenses are higher than your income this month."
            else:
                response += "Your income and expenses are balanced this month."

            return response

        return "I don't have enough monthly data yet."

    balance = float(total_income) - float(total_expense)

    if total_income == 0 and total_expense == 0:
        return "You don't have enough financial data yet. Add more transactions so I can give better advice."

    return f"Based on your current data, your income is {float(total_income):.2f}, your expense is {float(total_expense):.2f}, and your balance is {balance:.2f}. You can ask me about spending, budget tips, or monthly analysis."


@app.route("/")
def dashboard():
    conn = get_db_connection()

    recent_transactions = conn.execute(
        "SELECT * FROM transactions ORDER BY date DESC, id DESC LIMIT 6"
    ).fetchall()

    total_expense = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='expense'"
    ).fetchone()[0]

    total_income = conn.execute(
        "SELECT IFNULL(SUM(amount), 0) FROM transactions WHERE type='income'"
    ).fetchone()[0]

    total_count = conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    expense_categories = conn.execute("""
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE type='expense'
        GROUP BY category
        ORDER BY total DESC
        LIMIT 5
    """).fetchall()

    conn.close()

    balance = float(total_income) - float(total_expense)

    chart_labels = [item["category"] for item in expense_categories] if expense_categories else []
    chart_values = [float(item["total"]) for item in expense_categories] if expense_categories else []

    insights = generate_ai_insights()
    monthly_insights = generate_monthly_ai_insights()

    return render_template(
        "dashboard.html",
        recent_transactions=recent_transactions,
        total_expense=float(total_expense),
        total_income=float(total_income),
        total_count=total_count,
        expense_categories=expense_categories,
        balance=balance,
        chart_labels=chart_labels,
        chart_values=chart_values,
        insights=insights,
        monthly_insights=monthly_insights
    )


@app.route("/scan", methods=["GET", "POST"])
def scan_receipt():
    if request.method == "POST":
        try:
            if "receipt" not in request.files:
                flash("File tidak ditemukan.")
                return redirect(url_for("scan_receipt"))

            file = request.files["receipt"]

            if file.filename == "":
                flash("Pilih gambar struk dulu.")
                return redirect(url_for("scan_receipt"))

            if not allowed_file(file.filename):
                flash("Format file harus PNG, JPG, JPEG, atau WEBP.")
                return redirect(url_for("scan_receipt"))

            filename = secure_filename(file.filename)
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(original_path)

            processed_path = preprocess_receipt_image(original_path)

            results = reader.readtext(processed_path, detail=0)
            text_lines = [x.strip() for x in results if x.strip()]

            if not text_lines:
                flash("Teks pada struk tidak terbaca.")
                return redirect(url_for("scan_receipt"))

            store_name = extract_store_name(text_lines)
            amount = extract_total(text_lines)
            date = extract_date(text_lines)
            category = suggest_category(store_name)

            return render_template(
                "scan_receipt.html",
                preview_mode=True,
                receipt_image=filename,
                ocr_lines=text_lines,
                store_name=store_name,
                amount=f"{amount:.2f}",
                date=date,
                category=category
            )

        except Exception as e:
            flash(f"Terjadi error saat scan: {str(e)}")
            return redirect(url_for("scan_receipt"))

    return render_template("scan_receipt.html", preview_mode=False)


@app.route("/save-scanned", methods=["POST"])
def save_scanned():
    try:
        store_name = request.form["store_name"]
        amount_raw = request.form["amount"].replace(",", ".")
        amount = float(amount_raw)
        category = request.form["category"]
        date = request.form["date"]
        note = request.form.get("note", "")
        receipt_image = request.form["receipt_image"]

        conn = get_db_connection()
        conn.execute("""
            INSERT INTO transactions (store_name, amount, category, note, date, type, receipt_image)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (store_name, amount, category, note, date, "expense", receipt_image))
        conn.commit()
        conn.close()

        flash("Receipt berhasil disimpan.")
        return redirect(url_for("dashboard"))

    except Exception as e:
        flash(f"Gagal menyimpan: {str(e)}")
        return redirect(url_for("scan_receipt"))


@app.route("/manual", methods=["GET", "POST"])
def manual_input():
    if request.method == "POST":
        try:
            transaction_type = request.form["type"]
            store_name = request.form["store_name"]
            amount = float(request.form["amount"].replace(",", "."))
            note = request.form.get("note", "")
            category = request.form["category"]
            date = request.form["date"]

            if category == "AUTO":
                category = ai_predict_category(store_name, note, transaction_type)

            conn = get_db_connection()
            conn.execute("""
                INSERT INTO transactions (store_name, amount, category, note, date, type, receipt_image)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (store_name, amount, category, note, date, transaction_type, None))
            conn.commit()
            conn.close()

            flash("Manual transaction berhasil ditambahkan.")
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Gagal menambahkan transaksi manual: {str(e)}")
            return redirect(url_for("manual_input"))

    return render_template("manual_input.html")


@app.route("/transactions")
def transactions():
    transaction_type = request.args.get("type", "").strip()
    category = request.args.get("category", "").strip()
    search = request.args.get("search", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()

    conn = get_db_connection()

    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if transaction_type:
        query += " AND type = ?"
        params.append(transaction_type)

    if category:
        query += " AND category = ?"
        params.append(category)

    if search:
        query += " AND (LOWER(store_name) LIKE ? OR LOWER(note) LIKE ?)"
        params.append(f"%{search.lower()}%")
        params.append(f"%{search.lower()}%")

    if date_from:
        query += " AND date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND date <= ?"
        params.append(date_to)

    query += " ORDER BY date DESC, id DESC"

    rows = conn.execute(query, params).fetchall()

    categories = conn.execute(
        "SELECT DISTINCT category FROM transactions ORDER BY category ASC"
    ).fetchall()

    conn.close()

    return render_template(
        "transactions.html",
        transactions=rows,
        categories=categories,
        current_type=transaction_type,
        current_category=category,
        current_search=search,
        current_date_from=date_from,
        current_date_to=date_to
    )


@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_transaction(id):
    conn = get_db_connection()
    transaction = conn.execute(
        "SELECT * FROM transactions WHERE id = ?",
        (id,)
    ).fetchone()

    if not transaction:
        conn.close()
        flash("Transaction not found.")
        return redirect(url_for("transactions"))

    if request.method == "POST":
        try:
            transaction_type = request.form["type"]
            store_name = request.form["store_name"]
            amount = float(request.form["amount"].replace(",", "."))
            category = request.form["category"]
            date = request.form["date"]
            note = request.form.get("note", "")

            conn.execute("""
                UPDATE transactions
                SET store_name = ?, amount = ?, category = ?, note = ?, date = ?, type = ?
                WHERE id = ?
            """, (store_name, amount, category, note, date, transaction_type, id))
            conn.commit()
            conn.close()

            flash("Transaction updated successfully.")
            return redirect(url_for("transactions"))

        except Exception as e:
            conn.close()
            flash(f"Failed to update transaction: {str(e)}")
            return redirect(url_for("edit_transaction", id=id))

    conn.close()
    return render_template("edit_transaction.html", transaction=transaction)


@app.route("/delete/<int:id>")
def delete_transaction(id):
    try:
        conn = get_db_connection()

        row = conn.execute(
            "SELECT receipt_image FROM transactions WHERE id = ?",
            (id,)
        ).fetchone()

        if row and row["receipt_image"]:
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], row["receipt_image"])
            processed_path = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + row["receipt_image"])

            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(processed_path):
                os.remove(processed_path)

        conn.execute("DELETE FROM transactions WHERE id = ?", (id,))
        conn.commit()
        conn.close()

        flash("Transaction dihapus.")
        return redirect(url_for("transactions"))

    except Exception as e:
        flash(f"Gagal menghapus: {str(e)}")
        return redirect(url_for("transactions"))


@app.route("/advisor", methods=["GET", "POST"])
def advisor():
    suggestions = [
        "How is my spending this month?",
        "What's my biggest expense?",
        "Give me budgeting tips",
        "Analyze my income vs expense"
    ]

    if "advisor_messages" not in session:
        session["advisor_messages"] = [
            {
                "role": "ai",
                "text": "Hello! I'm your AI Financial Advisor. I can help you understand your spending habits, provide budgeting tips, and answer questions about your finances. How can I assist you today?"
            }
        ]

    messages = session["advisor_messages"]

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()

        if user_message:
            messages.append({
                "role": "user",
                "text": user_message
            })

            ai_reply = generate_chat_advice(user_message)

            messages.append({
                "role": "ai",
                "text": ai_reply
            })

            session["advisor_messages"] = messages
            session.modified = True

        return redirect(url_for("advisor"))

    return render_template(
        "advisor.html",
        suggestions=suggestions,
        messages=messages
    )


@app.route("/advisor/reset", methods=["POST"])
def reset_advisor():
    session.pop("advisor_messages", None)
    flash("AI Advisor chat has been reset.")
    return redirect(url_for("advisor"))


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
