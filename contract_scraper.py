from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

driver.get("https://www.incometaxindia.gov.in/indian-contract-act-1872")

time.sleep(5)

sections_data = []

for i in range(10):  # keep small for testing
    try:
        # Scroll
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        sections = driver.find_elements(By.XPATH, "//div[contains(@class,'card')]")

        if i >= len(sections):
            break

        section = sections[i]

        title = section.text
        print(f"\nOpening: {title}")

        driver.execute_script("arguments[0].click();", section)
        time.sleep(4)

        # 🔥 FIXED CONTENT SELECTOR
        content = driver.find_element(By.XPATH, "//div[contains(@class,'tab-content')]").text

        sections_data.append(f"{title}\n\n{content}\n\n{'-'*60}\n")

        time.sleep(2)

    except Exception as e:
        print("Error:", e)
        continue

driver.quit()

with open("contract_act_full.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sections_data))

print("\n✅ Data saved to contract_act_full.txt")