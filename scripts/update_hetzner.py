import psycopg2

conn = None
cur = conn.cursor()

updates = [
    "UPDATE facilities SET capacity = 438 WHERE id = 2398",
    "UPDATE facilities SET capacity = 159 WHERE id = 654",
    "UPDATE facilities SET capacity = 313 WHERE id = 3110",
    "UPDATE facilities SET capacity = 440 WHERE id = 6731",
    "UPDATE facilities SET capacity = 250 WHERE id = 5282",
    "UPDATE facilities SET capacity = 240 WHERE id = 258",
    "UPDATE facilities SET capacity = 120 WHERE id = 3113",
    "UPDATE facilities SET capacity = 65 WHERE id = 3116",
    "UPDATE facilities SET capacity = 44 WHERE id = 1906",
    "UPDATE facilities SET capacity = 36 WHERE id = 5277",
    "UPDATE facilities SET capacity = 30 WHERE id = 3359",
    "UPDATE facilities SET capacity = 50 WHERE id = 1486",
    "UPDATE facilities SET capacity = 80 WHERE id = 742",
    "UPDATE facilities SET capacity = 50 WHERE id = 71",
    "UPDATE facilities SET capacity = 60 WHERE id = 2913",
    "UPDATE facilities SET capacity = 30 WHERE id = 2498",
    "UPDATE facilities SET capacity = 50 WHERE id = 3361",
    "UPDATE facilities SET capacity = 20 WHERE id = 1547",
    "UPDATE facilities SET capacity = 350 WHERE facility_type = 'school' AND status = 'existing' AND capacity = 0",
    "UPDATE facilities SET capacity = 2000 WHERE facility_type = 'high_school' AND status = 'existing' AND capacity = 0",
    "UPDATE facilities SET capacity = 1800 WHERE facility_type = 'health_center' AND status = 'existing' AND capacity = 0",
]

for sql in updates:
    cur.execute(sql)
    print(f"OK ({cur.rowcount} rows): {sql[:70]}")

conn.commit()
cur.close()
conn.close()
print("Completado.")
