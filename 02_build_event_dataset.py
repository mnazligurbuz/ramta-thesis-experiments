# -*- coding: utf-8 -*-
"""
Step 2: Build the political event dataset.
Curated set of real, dated political events (2015-2025) with a short
discourse text (statement / announcement summary) and category.
Labels (market direction / magnitude) are computed from REAL market data
by joining with financial_data.csv over t+1, t+3, t+7 windows.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# (date, category, actor, text)
EVENTS = [
 ("2015-01-22","Monetary Policy","ECB / Mario Draghi","The ECB announces an expanded asset purchase programme of 60 billion euros per month, committing to purchases until inflation returns towards target."),
 ("2015-06-27","Geopolitical","Greek Government / Alexis Tsipras","Greek PM Tsipras announces a surprise referendum on the bailout terms proposed by creditors, raising the prospect of Greek exit from the eurozone."),
 ("2015-07-05","Election","Greek Referendum","Greek voters reject the bailout terms with a decisive No vote, deepening uncertainty over Greece's future in the euro area."),
 ("2015-08-11","Policy Announce","People's Bank of China","China devalues the yuan by nearly 2 percent, the largest single-day move in decades, framing it as a market-oriented reform."),
 ("2015-11-13","Geopolitical","French Government","Coordinated terrorist attacks strike Paris; President Hollande declares a state of emergency and announces France is at war with ISIS."),
 ("2015-12-16","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve raises interest rates for the first time in nearly a decade, signalling confidence in the US recovery while stressing a gradual path."),
 ("2016-02-20","Election","UK Government / David Cameron","PM Cameron announces the date of the EU membership referendum, launching the official Brexit campaign period."),
 ("2016-06-24","Election","UK Referendum Result","The UK votes to leave the European Union; PM Cameron announces his resignation, triggering historic uncertainty over UK-EU relations."),
 ("2016-07-11","Policy Announce","UK Government","Theresa May becomes the new UK Prime Minister, pledging that Brexit means Brexit and promising negotiation clarity."),
 ("2016-11-09","Election","US Presidential Election","Donald Trump wins the US presidential election, promising sweeping changes to trade policy, taxation, and international agreements."),
 ("2016-12-04","Election","Italian Referendum / Matteo Renzi","Italian voters reject constitutional reform; PM Renzi announces his resignation, raising concerns over Italian political stability and banks."),
 ("2017-01-20","Policy Announce","US Government / Donald Trump","President Trump is inaugurated, declaring an America First agenda on trade, immigration, and foreign policy in his inaugural address."),
 ("2017-03-29","Geopolitical","UK Government / Theresa May","The UK formally triggers Article 50, starting the two-year countdown to withdrawal from the European Union."),
 ("2017-04-24","Election","French Presidential Election","Emmanuel Macron and Marine Le Pen advance to the French presidential runoff; markets weigh pro-EU continuity against eurosceptic risk."),
 ("2017-05-08","Election","French Presidential Election","Emmanuel Macron wins the French presidency decisively, easing fears of French exit from the euro and boosting European integration prospects."),
 ("2017-08-09","Geopolitical","US Government / Donald Trump","President Trump warns North Korea it will face fire and fury like the world has never seen, escalating nuclear tensions."),
 ("2017-09-25","Election","German Federal Election","Chancellor Merkel wins a fourth term but with a weakened coalition, as the far-right AfD enters the Bundestag for the first time."),
 ("2017-10-02","Geopolitical","Catalan Government","Catalonia holds an independence referendum declared illegal by Madrid; violent scenes raise Spanish political risk."),
 ("2017-12-20","Policy Announce","US Congress","The US Congress passes the Tax Cuts and Jobs Act, the largest corporate tax overhaul in decades, cutting the corporate rate to 21 percent."),
 ("2018-03-01","Policy Announce","US Government / Donald Trump","President Trump announces tariffs of 25 percent on steel and 10 percent on aluminum imports, citing national security grounds."),
 ("2018-03-22","Policy Announce","US Government / Donald Trump","President Trump signs a memorandum targeting up to 60 billion dollars of Chinese imports with tariffs, igniting fears of a US-China trade war."),
 ("2018-05-08","Geopolitical","US Government / Donald Trump","President Trump announces US withdrawal from the Iran nuclear deal and the reimposition of sanctions on Iran."),
 ("2018-06-12","Geopolitical","US-North Korea Summit","President Trump and Kim Jong-un hold a historic summit in Singapore, signing a joint statement on denuclearisation of the Korean peninsula."),
 ("2018-07-06","Policy Announce","US Government","The first tranche of US tariffs on 34 billion dollars of Chinese goods takes effect; China retaliates immediately with equivalent measures."),
 ("2018-09-17","Policy Announce","US Government / Donald Trump","President Trump announces 10 percent tariffs on an additional 200 billion dollars of Chinese imports, threatening escalation to 25 percent."),
 ("2018-11-25","Geopolitical","EU / UK Governments","EU leaders endorse the Brexit withdrawal agreement negotiated with PM May, moving the deal to a high-stakes UK parliamentary vote."),
 ("2018-12-03","Policy Announce","US-China G20 Meeting","Presidents Trump and Xi agree a 90-day truce in the trade war at the G20 in Buenos Aires, delaying planned tariff increases."),
 ("2018-12-19","Monetary Policy","Federal Reserve / Jerome Powell","The Fed raises rates a fourth time in 2018 and signals further tightening, disappointing markets hoping for a dovish pivot."),
 ("2019-01-15","Election","UK Parliament","The UK Parliament rejects PM May's Brexit deal by a historic margin of 230 votes, deepening uncertainty over the withdrawal process."),
 ("2019-05-05","Policy Announce","US Government / Donald Trump","President Trump tweets that tariffs on 200 billion dollars of Chinese goods will rise to 25 percent, abruptly ending the trade truce."),
 ("2019-07-24","Policy Announce","UK Government","Boris Johnson becomes UK Prime Minister, pledging to deliver Brexit by October 31 with or without a deal."),
 ("2019-07-31","Monetary Policy","Federal Reserve / Jerome Powell","The Fed cuts interest rates for the first time since 2008, describing the move as a mid-cycle adjustment rather than the start of an easing cycle."),
 ("2019-08-01","Policy Announce","US Government / Donald Trump","President Trump announces 10 percent tariffs on the remaining 300 billion dollars of Chinese imports, escalating the trade war."),
 ("2019-08-23","Policy Announce","US Government / Donald Trump","President Trump orders US companies to seek alternatives to China after Beijing announces retaliatory tariffs, roiling markets."),
 ("2019-10-11","Policy Announce","US-China Negotiations","President Trump announces a Phase One trade deal in principle with China, suspending a planned tariff increase."),
 ("2019-12-13","Election","UK General Election","PM Johnson wins a large Conservative majority, securing a mandate to get Brexit done by the end of January."),
 ("2020-01-03","Geopolitical","US Government","A US drone strike kills Iranian General Qassem Soleimani in Baghdad; Iran vows severe revenge, spiking geopolitical risk."),
 ("2020-01-15","Policy Announce","US-China Governments","The US and China sign the Phase One trade agreement, committing China to increased purchases of US goods."),
 ("2020-01-31","Geopolitical","UK Government","The United Kingdom formally leaves the European Union, entering an 11-month transition period."),
 ("2020-03-11","Geopolitical","WHO / Governments","The WHO declares COVID-19 a pandemic; governments worldwide begin announcing emergency containment measures."),
 ("2020-03-13","Policy Announce","US Government / Donald Trump","President Trump declares a national emergency over COVID-19, unlocking federal funds and signalling major fiscal response."),
 ("2020-03-15","Monetary Policy","Federal Reserve / Jerome Powell","In an emergency Sunday move, the Fed cuts rates to near zero and announces 700 billion dollars of quantitative easing."),
 ("2020-03-26","Policy Announce","US Congress","The US Senate passes the 2 trillion dollar CARES Act, the largest fiscal stimulus in American history."),
 ("2020-07-21","Policy Announce","European Council","EU leaders agree the 750 billion euro Next Generation EU recovery fund, a landmark step towards common European borrowing."),
 ("2020-11-07","Election","US Presidential Election","Major networks declare Joe Biden winner of the US presidential election; President Trump refuses to concede."),
 ("2020-12-24","Geopolitical","EU / UK Governments","The EU and UK announce a Trade and Cooperation Agreement, averting a no-deal Brexit days before the transition deadline."),
 ("2021-01-06","Geopolitical","US Political Crisis","A mob storms the US Capitol as Congress certifies the election result; President Trump's remarks are blamed for inciting the violence."),
 ("2021-03-11","Policy Announce","US Government / Joe Biden","President Biden signs the 1.9 trillion dollar American Rescue Plan, promising direct payments and accelerated vaccination."),
 ("2021-08-15","Geopolitical","Afghan Government Collapse","Kabul falls to the Taliban as the US completes its withdrawal; President Biden defends the decision amid chaotic evacuation scenes."),
 ("2021-09-26","Election","German Federal Election","Germany's SPD narrowly wins the federal election; Chancellor Merkel's era ends with months of coalition talks ahead."),
 ("2021-11-22","Monetary Policy","US Government / Joe Biden","President Biden renominates Jerome Powell as Fed Chair, signalling continuity in US monetary policy leadership."),
 ("2022-02-21","Geopolitical","Russian Government / Vladimir Putin","President Putin recognises the separatist Donetsk and Luhansk republics and orders troops into eastern Ukraine."),
 ("2022-02-24","Geopolitical","Russian Government / Vladimir Putin","Russia launches a full-scale invasion of Ukraine; President Putin announces a special military operation in a televised address."),
 ("2022-02-26","Policy Announce","Western Governments","The US, EU and allies announce the removal of key Russian banks from SWIFT and sanctions on Russia's central bank."),
 ("2022-03-08","Policy Announce","US Government / Joe Biden","President Biden announces a ban on Russian oil, gas and energy imports to the United States."),
 ("2022-03-16","Monetary Policy","Federal Reserve / Jerome Powell","The Fed raises rates for the first time since 2018, beginning what it signals will be a sustained tightening cycle against inflation."),
 ("2022-06-15","Monetary Policy","Federal Reserve / Jerome Powell","The Fed hikes rates by 75 basis points, the largest single increase since 1994, and signals more aggressive action against inflation."),
 ("2022-07-21","Monetary Policy","ECB / Christine Lagarde","The ECB raises rates for the first time in 11 years with a larger-than-expected 50 basis point hike and unveils an anti-fragmentation tool."),
 ("2022-08-26","Monetary Policy","Federal Reserve / Jerome Powell","At Jackson Hole, Chair Powell warns that fighting inflation will bring pain to households and that the Fed must keep at it until the job is done."),
 ("2022-09-21","Geopolitical","Russian Government / Vladimir Putin","President Putin announces partial military mobilisation and issues veiled nuclear threats, escalating the war in Ukraine."),
 ("2022-09-23","Policy Announce","UK Government / Kwasi Kwarteng","The UK mini-budget announces the largest unfunded tax cuts in 50 years, triggering a collapse in sterling and gilt market turmoil."),
 ("2022-10-20","Policy Announce","UK Government / Liz Truss","PM Liz Truss resigns after 44 days in office following the market chaos caused by the mini-budget."),
 ("2023-01-08","Geopolitical","Brazilian Political Crisis","Supporters of former President Bolsonaro storm Brazil's Congress, Supreme Court and presidential palace."),
 ("2023-03-19","Policy Announce","Swiss Government / Regulators","Swiss authorities announce the emergency takeover of Credit Suisse by UBS, containing a banking confidence crisis with political backing."),
 ("2023-05-27","Policy Announce","US Government","President Biden and Speaker McCarthy announce a debt ceiling agreement, averting a US default days before the deadline."),
 ("2023-07-26","Monetary Policy","Federal Reserve / Jerome Powell","The Fed raises rates to a 22-year high; Chair Powell keeps options open, saying future decisions depend on incoming data."),
 ("2023-10-07","Geopolitical","Hamas / Israeli Government","Hamas launches a large-scale attack on Israel; PM Netanyahu declares war, opening a major new Middle East conflict."),
 ("2023-10-27","Geopolitical","Israeli Government","Israel launches its ground offensive in Gaza, expanding the war amid international calls for restraint."),
 ("2024-04-13","Geopolitical","Iranian Government","Iran launches a direct drone and missile attack on Israel, the first attack from Iranian soil, raising fears of regional war."),
 ("2024-06-09","Election","French Government / Emmanuel Macron","President Macron dissolves the National Assembly and calls snap elections after a far-right surge in European elections."),
 ("2024-07-21","Election","US Government / Joe Biden","President Biden withdraws from the presidential race and endorses Vice President Harris, upending the US election campaign."),
 ("2024-09-18","Monetary Policy","Federal Reserve / Jerome Powell","The Fed begins its easing cycle with an outsized 50 basis point rate cut, declaring greater confidence that inflation is moving to target."),
 ("2024-11-06","Election","US Presidential Election","Donald Trump wins the US presidential election decisively, promising universal tariffs, tax cuts, and mass deportations."),
 ("2024-12-04","Geopolitical","South Korean Government","South Korea's President Yoon declares martial law in a shock late-night address, before parliament votes to overturn it within hours."),
 ("2025-01-20","Policy Announce","US Government / Donald Trump","President Trump is inaugurated for a second term, signing a wave of executive orders on trade, energy and immigration."),
 ("2025-02-23","Election","German Federal Election","Germany holds federal elections; the CDU/CSU wins as the far-right AfD doubles its vote share, complicating coalition building."),
 ("2025-04-02","Policy Announce","US Government / Donald Trump","President Trump announces sweeping Liberation Day reciprocal tariffs on nearly all trading partners, shocking global markets."),
 ("2025-04-09","Policy Announce","US Government / Donald Trump","President Trump announces a 90-day pause on most reciprocal tariffs hours after they took effect, citing negotiations."),
 ("2025-06-13","Geopolitical","Israeli Government","Israel launches strikes on Iranian nuclear and military sites; Iran retaliates, sharply escalating the regional conflict."),

 # ── Federal Reserve (FOMC) policy decisions — well-documented, high-confidence ──
 ("2016-12-14","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve raises interest rates for the second time since the crisis, citing labour market strength and rising confidence in the economic outlook."),
 ("2017-03-15","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve raises interest rates a third time as the recovery firms, with officials signalling a gradual pace of further increases."),
 ("2017-06-14","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve raises rates again and announces details of its plan to begin shrinking its balance sheet later in the year."),
 ("2017-12-13","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve raises rates in Chair Yellen's final meeting, while forecasting three more hikes in 2018 as tax cuts boost the outlook."),
 ("2018-03-21","Monetary Policy","Federal Reserve / Jerome Powell","In his first meeting as Chair, Jerome Powell's Federal Reserve raises rates and signals a slightly steeper hiking path for the year ahead."),
 ("2018-06-13","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates and revises its projections to show four hikes in total for 2018, up from three."),
 ("2018-09-26","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates for the third time in 2018 and removes language describing policy as accommodative."),
 ("2019-09-18","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve cuts rates for a second consecutive meeting amid trade-war uncertainty, while Chair Powell resists commitment to further cuts."),
 ("2019-10-30","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve cuts rates a third time and signals a pause, saying current policy is likely to remain appropriate absent a material change in outlook."),
 ("2020-03-03","Monetary Policy","Federal Reserve / Jerome Powell","In an unscheduled emergency move, the Federal Reserve cuts rates by half a point, its first inter-meeting cut since the 2008 crisis, as COVID-19 spreads."),
 ("2021-11-03","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve announces it will begin tapering its pandemic-era bond purchases, a first step toward policy normalisation."),
 ("2021-12-15","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve accelerates the pace of taper amid persistent inflation and pencils in three rate hikes for 2022."),
 ("2022-05-04","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates by half a point, its largest single hike since 2000, and announces the start of balance-sheet reduction."),
 ("2022-07-27","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates by 75 basis points for a second consecutive meeting as it races to contain persistently high inflation."),
 ("2022-11-02","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates by 75 basis points for a fourth straight meeting, while Chair Powell hints at a slower future pace."),
 ("2022-12-14","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve downshifts to a 50 basis point hike but projects rates will peak higher than previously expected in 2023."),
 ("2023-02-01","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve slows further to a 25 basis point hike, with Chair Powell acknowledging that the disinflation process has begun."),
 ("2023-03-22","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates by 25 basis points despite recent banking-sector turmoil following the collapse of Silicon Valley Bank."),
 ("2023-05-03","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve raises rates again and signals a possible pause, removing prior guidance about the need for additional firming."),
 ("2023-09-20","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds rates steady but signals a higher-for-longer stance, with officials projecting one more hike in 2023."),
 ("2023-12-13","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds rates steady and the dot plot signals three rate cuts in 2024, sparking a rally across risk assets."),
 ("2024-07-31","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds rates steady while Chair Powell opens the door to a rate cut as soon as the September meeting."),
 ("2024-11-07","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve cuts rates by 25 basis points, its second cut of the cycle, days after Donald Trump's election victory."),
 ("2024-12-18","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve cuts rates by 25 basis points but signals a slower pace of cuts in 2025, adopting a more hawkish tone."),

 # ── European Central Bank (ECB) policy decisions ─────────────────────────────
 ("2016-03-10","Monetary Policy","ECB / Mario Draghi","The ECB cuts its deposit rate deeper into negative territory, cuts the refi rate to zero, and expands its asset purchase programme."),
 ("2019-09-12","Monetary Policy","ECB / Mario Draghi","In one of his final major decisions as President, Mario Draghi's ECB cuts the deposit rate and relaunches open-ended quantitative easing."),
 ("2021-12-16","Monetary Policy","ECB / Christine Lagarde","The ECB announces it will end its pandemic emergency purchase programme in March 2022, a first step toward withdrawing crisis-era stimulus."),
 ("2023-02-02","Monetary Policy","ECB / Christine Lagarde","The ECB raises rates by 50 basis points and signals another 50 basis point increase at its next meeting to fight persistent inflation."),
 ("2023-05-04","Monetary Policy","ECB / Christine Lagarde","The ECB slows the pace of tightening to a 25 basis point hike as inflation pressures begin to moderate."),
 ("2023-09-14","Monetary Policy","ECB / Christine Lagarde","The ECB raises rates by 25 basis points to a record high and signals that the peak of the tightening cycle has likely been reached."),
 ("2023-10-26","Monetary Policy","ECB / Christine Lagarde","The ECB holds interest rates steady for the first time in over a year, formally ending its aggressive tightening cycle."),
 ("2024-06-06","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates for the first time since 2019, moving ahead of the Federal Reserve in beginning its easing cycle."),
 ("2024-09-12","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates for a second time in 2024 as inflation continues to ease across the eurozone."),
 ("2024-10-17","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates for a third time, accelerating its pace of easing amid growing concerns over eurozone growth."),
 ("2024-12-12","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates for a fourth time in 2024, continuing its steady easing cycle as inflation nears target."),
 ("2025-01-30","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates again, continuing its gradual easing cycle as growth in the eurozone remains sluggish."),
 ("2025-03-06","Monetary Policy","ECB / Christine Lagarde","The ECB cuts interest rates further while President Lagarde warns that escalating US tariff threats pose a significant risk to the eurozone outlook."),

 # ── Bank of England (BoE) policy decisions ───────────────────────────────────
 ("2016-08-04","Monetary Policy","BoE / Mark Carney","The Bank of England cuts interest rates to a record low of 0.25% and expands its asset purchase programme, its first stimulus package since the Brexit referendum."),
 ("2021-12-16","Monetary Policy","BoE / Andrew Bailey","The Bank of England becomes the first major central bank to raise interest rates since the pandemic began, surprising markets that had expected a hold."),
 ("2022-09-28","Monetary Policy","BoE / Andrew Bailey","The Bank of England launches emergency bond purchases to calm a collapsing gilt market in the wake of the government's mini-budget."),
 ("2023-08-03","Monetary Policy","BoE / Andrew Bailey","The Bank of England raises interest rates to 5.25%, a fourteenth consecutive increase, while signalling that the tightening cycle is nearing its peak."),
 ("2024-08-01","Monetary Policy","BoE / Andrew Bailey","The Bank of England cuts interest rates for the first time since the pandemic began, narrowly voting to ease policy as inflation cools."),
 ("2024-11-07","Monetary Policy","BoE / Andrew Bailey","The Bank of England cuts interest rates for a second time in 2024, continuing a cautious and gradual easing cycle."),

 # ── Bank of Japan (BoJ) policy decisions ─────────────────────────────────────
 ("2016-01-29","Monetary Policy","BoJ / Haruhiko Kuroda","The Bank of Japan adopts a negative interest rate policy for the first time in its history, unsettling markets before the yen rebounds sharply."),
 ("2022-09-22","Policy Announce","Japanese Government / Ministry of Finance","Japan intervenes directly in currency markets to support the yen for the first time since 1998, after USD/JPY breaches the 145 level."),
 ("2024-03-19","Monetary Policy","BoJ / Kazuo Ueda","The Bank of Japan ends its negative interest rate policy and yield curve control programme, delivering its first rate hike since 2007."),
 ("2024-07-31","Monetary Policy","BoJ / Kazuo Ueda","The Bank of Japan raises interest rates again and signals further tightening ahead, surprising markets already unsettled by a strengthening yen."),
 ("2024-08-05","Geopolitical","Global Financial Markets","Global equity markets suffer one of their sharpest single-day selloffs in years as a rapid unwinding of the yen carry trade triggers a historic crash in Japanese stocks."),

 # ── Major elections (additional) ─────────────────────────────────────────────
 ("2018-10-28","Election","Brazilian Presidential Election","Jair Bolsonaro wins Brazil's presidential election on a market-friendly, pro-reform platform, boosting Brazilian assets."),
 ("2019-05-23","Election","Indian General Election","Narendra Modi's BJP wins a landslide re-election in India's general election, reinforcing expectations of policy continuity."),
 ("2022-10-30","Election","Brazilian Presidential Election","Luiz Inacio Lula da Silva narrowly defeats incumbent Jair Bolsonaro in Brazil's presidential runoff, raising uncertainty over fiscal policy direction."),
 ("2024-06-04","Election","Indian General Election","Narendra Modi's BJP unexpectedly loses its outright parliamentary majority, forcing a coalition government and triggering a sharp selloff in Indian equities."),
 ("2024-10-27","Election","Japanese General Election","Japan's ruling LDP loses its parliamentary majority in a snap election under new PM Shigeru Ishiba, raising domestic political uncertainty."),

 # ── Additional trade and geopolitical events ─────────────────────────────────
 ("2018-06-09","Geopolitical","G7 Summit","The G7 summit ends in open discord as President Trump refuses to sign the joint communique, escalating trade tensions with US allies."),
 ("2019-05-30","Policy Announce","US Government / Donald Trump","President Trump threatens escalating tariffs on all Mexican goods unless Mexico curbs illegal immigration, rattling markets before the threat is later suspended."),
 ("2019-06-29","Policy Announce","US-China G20 Meeting","Presidents Trump and Xi meet at the G20 Osaka summit and agree to resume trade talks, pausing further tariff escalation."),
 ("2021-01-20","Policy Announce","US Government / Joe Biden","Joe Biden is inaugurated as US President, signalling a return to multilateral diplomacy and a reversal of several Trump-era trade and climate policies."),
 ("2023-08-24","Policy Announce","BRICS Summit","The BRICS bloc announces a major expansion inviting six new member states, signalling a shift in global economic alignment."),

 # ── Swiss National Bank (SNB) ─────────────────────────────────────────────────
 ("2015-01-15","Monetary Policy","SNB / Thomas Jordan","The Swiss National Bank unexpectedly abandons its franc-euro exchange rate cap, causing a historic surge in the franc and severe disruption across global currency markets."),
 ("2022-06-16","Monetary Policy","SNB / Thomas Jordan","The Swiss National Bank surprises markets with its first interest rate hike in 15 years, ending its negative interest rate era earlier than expected."),

 # ── People's Bank of China (PBOC) ──────────────────────────────────────────────
 ("2018-06-24","Monetary Policy","PBOC / Chinese Government","The People's Bank of China cuts the reserve requirement ratio for major banks to support the economy amid escalating trade tensions with the United States."),
 ("2023-08-15","Monetary Policy","PBOC / Chinese Government","The People's Bank of China cuts key policy rates as a slew of weak economic data signals a deepening slowdown in the world's second-largest economy."),

 # ── OPEC+ production decisions ────────────────────────────────────────────────
 ("2016-11-30","Policy Announce","OPEC","OPEC agrees its first oil production cut deal in eight years, sending crude oil prices sharply higher and boosting commodity-linked currencies."),
 ("2020-03-08","Geopolitical","Saudi Arabia / Russia","Saudi Arabia launches an oil price war with Russia after OPEC+ production-cut talks collapse, triggering a historic crash in oil prices."),
 ("2020-04-12","Policy Announce","OPEC+","OPEC+ agrees a record production cut of nearly 10 million barrels per day in an emergency deal to support prices amid collapsing pandemic-era demand."),
 ("2022-10-05","Policy Announce","OPEC+","OPEC+ announces a major production cut of two million barrels per day, defying pressure from the United States ahead of the midterm elections."),

 # ── Banking sector crisis (2023) ────────────────────────────────────────────────
 ("2023-03-10","Policy Announce","US Regulators / Silicon Valley Bank","Silicon Valley Bank collapses and is taken over by regulators in the largest US bank failure since the 2008 financial crisis, triggering fears of wider banking contagion."),
 ("2023-03-12","Policy Announce","US Regulators / Federal Reserve","US regulators announce emergency measures to guarantee all Silicon Valley Bank deposits and launch a new Federal Reserve lending facility to stem banking-sector contagion."),

 # ── Additional Federal Reserve forward-guidance pivots ──────────────────────────
 ("2019-01-30","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve signals a 'patient' pause on further rate hikes, a sharp dovish pivot after the December 2018 hike had unsettled markets."),
 ("2021-06-16","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve's updated projections shift markedly more hawkish, pulling forward the expected timing of future rate hikes and surprising markets."),
 ("2022-01-26","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve signals an imminent March rate hike and confirms the end of its bond-buying programme, a decisively hawkish pivot amid persistent inflation."),

 # ── COVID-19 vaccine milestones (major market-moving events) ────────────────────
 ("2020-11-09","Policy Announce","Pfizer / BioNTech","Pfizer and BioNTech announce that their COVID-19 vaccine candidate is more than 90 percent effective in trials, triggering a sharp global equity market rally."),

 # ── Turkish presidential election (2023) ─────────────────────────────────────────
 ("2023-05-14","Election","Turkish Presidential Election","Turkey's presidential election proceeds to a runoff as incumbent Recep Tayyip Erdogan fails to secure an outright majority, keeping the lira under pressure."),
 ("2023-05-28","Election","Turkish Presidential Election","Recep Tayyip Erdogan wins Turkey's presidential runoff election, extending his rule into a third decade amid ongoing concerns over economic policy direction."),

 # ── German defence policy shift ("Zeitenwende") ──────────────────────────────────
 ("2022-02-27","Policy Announce","German Government / Olaf Scholz","Chancellor Scholz announces a historic increase in German defence spending and a special 100 billion euro fund, describing the shift as a Zeitenwende in response to Russia's invasion of Ukraine."),

 # ── Bank of Canada (BoC) ────────────────────────────────────────────────────────
 ("2015-01-21","Monetary Policy","BoC / Stephen Poloz","The Bank of Canada unexpectedly cuts interest rates, citing the sharp collapse in oil prices as a significant risk to the Canadian economy."),
 ("2022-03-02","Monetary Policy","BoC / Tiff Macklem","The Bank of Canada raises interest rates for the first time since the pandemic began, joining the global tightening cycle."),
 ("2022-07-13","Monetary Policy","BoC / Tiff Macklem","The Bank of Canada delivers a surprise full percentage point rate hike, its largest single increase since 1998, to combat persistent inflation."),

 # ── Greek debt crisis (additional) ────────────────────────────────────────────
 ("2015-06-29","Policy Announce","Greek Government","Greece imposes capital controls and closes its banks ahead of the bailout referendum as fears of a Greek exit from the euro intensify."),
 ("2015-07-13","Policy Announce","Eurozone Summit","Greece and its eurozone creditors reach an agreement on a third bailout programme after an all-night emergency summit, averting an imminent Greek exit from the euro."),

 # ── US midterm elections ───────────────────────────────────────────────────────
 ("2018-11-06","Election","US Midterm Elections","Democrats regain control of the US House of Representatives in the midterm elections, setting up a divided Congress for the remainder of the Trump presidency."),
 ("2022-11-08","Election","US Midterm Elections","Republicans narrowly win control of the US House in the midterm elections while Democrats retain the Senate, resulting in a closely divided Congress."),

 # ── Global supply-chain shock ─────────────────────────────────────────────────
 ("2021-03-23","Geopolitical","Suez Canal Authority","The container ship Ever Given runs aground and blocks the Suez Canal for six days, disrupting an estimated ten percent of global trade and roiling shipping and commodity markets."),

 # ── China Evergrande property crisis ──────────────────────────────────────────
 ("2021-09-20","Policy Announce","China Evergrande Group","Heavily indebted property developer China Evergrande misses bond payments, triggering fears of contagion across China's real estate sector and broader financial markets."),
 ("2023-08-17","Policy Announce","China Evergrande Group","China Evergrande files for bankruptcy protection in the United States, marking a new phase in China's prolonged property sector crisis."),

 # ── Turkish lira crisis (2018) ────────────────────────────────────────────────
 ("2018-08-10","Geopolitical","US Government / Donald Trump","President Trump announces a doubling of tariffs on Turkish steel and aluminium amid a diplomatic dispute, sharply deepening a currency crisis in Turkey."),
 ("2018-08-13","Policy Announce","Turkish Government / Central Bank of Turkey","Turkish authorities announce measures to support the lira and shore up bank liquidity, though investor confidence remains fragile amid the currency crisis."),

 # ── Brexit process (additional milestones) ────────────────────────────────────
 ("2019-10-17","Policy Announce","EU / UK Governments","The European Union and the United Kingdom agree a revised Brexit withdrawal agreement under new Prime Minister Boris Johnson, replacing the deal previously rejected by Parliament."),
 ("2020-12-31","Geopolitical","EU / UK Governments","The Brexit transition period ends at midnight, with the United Kingdom fully exiting the European Union single market and customs union."),

 # ── Historic market-wide selloffs ──────────────────────────────────────────────
 ("2015-08-24","Geopolitical","Global Financial Markets","Global stock markets suffer a sharp selloff dubbed 'Black Monday', driven by fears over a sharper-than-expected slowdown in the Chinese economy."),
 ("2018-02-05","Geopolitical","Global Financial Markets","US stock markets suffer their sharpest single-day point decline on record at the time, as fears of accelerating inflation and faster rate hikes trigger a volatility spike."),
 ("2020-02-24","Geopolitical","Global Financial Markets","Global equity markets begin a sharp selloff as COVID-19 spreads beyond China, marking the onset of the pandemic-driven market crash."),

 # ── Additional Federal Reserve pivots ───────────────────────────────────────────
 ("2015-09-17","Monetary Policy","Federal Reserve / Janet Yellen","The Federal Reserve unexpectedly holds interest rates steady, citing concerns over global economic and financial developments, notably the slowdown in China."),
 ("2020-03-23","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve announces open-ended quantitative easing and a series of emergency lending facilities to stabilise financial markets during the pandemic crash."),
 ("2021-08-27","Monetary Policy","Federal Reserve / Jerome Powell","At the Jackson Hole symposium, Chair Powell signals that the Fed could begin tapering asset purchases before year-end while stressing that rate hikes remain further off."),

 # ── Reserve Bank of Australia (RBA) ──────────────────────────────────────────
 ("2016-08-02","Monetary Policy","RBA / Glenn Stevens","The Reserve Bank of Australia cuts interest rates to a fresh record low, citing subdued inflation and the need to support the economy's transition away from mining investment."),
 ("2020-11-03","Monetary Policy","RBA / Philip Lowe","The Reserve Bank of Australia cuts its cash rate to a record low of 0.10% and launches a bond-buying programme to support the pandemic recovery."),
 ("2022-05-03","Monetary Policy","RBA / Philip Lowe","The Reserve Bank of Australia raises interest rates for the first time in over a decade, joining the global tightening cycle against inflation."),

 # ── Iran nuclear deal ─────────────────────────────────────────────────────────
 ("2015-07-14","Policy Announce","Iran / P5+1 Nations","Iran and world powers reach a landmark nuclear agreement, the JCPOA, agreeing to lift sanctions in exchange for curbs on Iran's nuclear programme."),
 ("2018-05-21","Geopolitical","US Government / Mike Pompeo","Secretary of State Pompeo outlines sweeping new demands on Iran following the US withdrawal from the nuclear deal, threatening the toughest sanctions in history."),

 # ── Hong Kong political unrest ────────────────────────────────────────────────
 ("2019-06-09","Geopolitical","Hong Kong Government","Mass protests erupt in Hong Kong against a proposed extradition bill, marking the start of a prolonged period of political unrest that unsettles regional markets."),
 ("2020-06-30","Policy Announce","Chinese Government","China imposes a sweeping national security law on Hong Kong, drawing international condemnation and raising concerns over the territory's autonomy and financial-hub status."),

 # ── Argentina and Mexico elections ────────────────────────────────────────────
 ("2023-11-19","Election","Argentine Presidential Election","Libertarian outsider Javier Milei wins Argentina's presidential runoff on a platform of radical economic reform, triggering a sharp rally in Argentine assets."),
 ("2024-06-02","Election","Mexican Presidential Election","Claudia Sheinbaum wins Mexico's presidential election in a landslide, becoming the country's first female president and continuing the ruling party's dominance."),

 # ── Red Sea shipping crisis ───────────────────────────────────────────────────
 ("2023-12-18","Geopolitical","Houthi Movement / Red Sea Shipping","Houthi attacks on commercial shipping in the Red Sea intensify, prompting major shipping firms to suspend transits and disrupting global trade routes."),
 ("2024-01-12","Geopolitical","US / UK Governments","The United States and United Kingdom launch military strikes against Houthi targets in Yemen in response to continued attacks on Red Sea shipping."),

 # ── Banking sector crisis (additional) ──────────────────────────────────────────
 ("2023-05-01","Policy Announce","US Regulators / First Republic Bank","First Republic Bank is seized by regulators and sold to JPMorgan Chase in the second-largest US bank failure in history, extending the regional banking crisis."),
 ("2023-03-24","Policy Announce","Deutsche Bank","Shares in Deutsche Bank plunge and its credit default swap costs spike sharply amid fears of wider contagion from the banking turmoil following Credit Suisse's rescue."),

 # ── Additional Federal Reserve meetings ─────────────────────────────────────────
 ("2016-11-02","Monetary Policy","Federal Reserve","The Federal Reserve holds interest rates steady just days before the US presidential election, avoiding any appearance of political interference."),
 ("2018-08-01","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds interest rates steady while describing the economy as growing at a strong pace, keeping alive expectations of further hikes later in the year."),
 ("2020-06-10","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve projects near-zero interest rates through 2022 and pledges continued asset purchases to support the pandemic recovery."),
 ("2023-06-14","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds interest rates steady for the first time in over a year, pausing its tightening cycle while signalling further hikes may still follow."),
 ("2024-05-01","Monetary Policy","Federal Reserve / Jerome Powell","The Federal Reserve holds interest rates steady and Chair Powell acknowledges that progress on inflation has stalled, dampening hopes for near-term rate cuts."),

 # ── Additional ECB meetings ────────────────────────────────────────────────────
 ("2015-12-03","Monetary Policy","ECB / Mario Draghi","The ECB extends and expands its quantitative easing programme but disappoints markets expecting more aggressive stimulus, triggering a sharp bond selloff."),
 ("2019-03-07","Monetary Policy","ECB / Mario Draghi","The ECB unveils a new round of cheap long-term loans for banks and pushes back the timeline for future rate hikes, citing a weakening growth outlook."),

 # ── UK and French elections (additional) ──────────────────────────────────────
 ("2017-06-08","Election","UK General Election","Prime Minister Theresa May's Conservative Party unexpectedly loses its parliamentary majority in a snap general election, weakening her position ahead of Brexit negotiations."),
 ("2022-04-24","Election","French Presidential Election","Emmanuel Macron wins re-election as French President, defeating far-right challenger Marine Le Pen in the runoff and easing fears of a eurosceptic shift in France."),

 # ── French pension reform protests ────────────────────────────────────────────
 ("2023-03-16","Policy Announce","French Government / Emmanuel Macron","The French government forces through a controversial pension reform raising the retirement age without a parliamentary vote, triggering nationwide protests."),

 # ── G7 summit and geopolitical crises (additional) ────────────────────────────
 ("2019-08-25","Geopolitical","G7 Summit","The G7 summit in Biarritz is overshadowed by trade tensions as President Trump reverses course multiple times on new tariffs against China."),
 ("2020-08-04","Geopolitical","Lebanese Government","A massive explosion devastates the port of Beirut, killing over 200 people and plunging Lebanon deeper into economic and political crisis."),

 # ── US-China trade war (additional) ────────────────────────────────────────────
 ("2018-12-01","Geopolitical","Canada / China Governments","Canada arrests Huawei CFO Meng Wanzhou at the request of the United States, sharply escalating tensions between China and the West amid the ongoing trade war."),
 ("2019-05-10","Policy Announce","US Government / Donald Trump","The Trump administration raises tariffs on 200 billion dollars of Chinese goods from 10 to 25 percent after trade talks stall, reigniting the trade war."),

 # ── Political corruption scandal with global market impact ───────────────────
 ("2016-04-03","Geopolitical","International Consortium of Investigative Journalists","Leaked 'Panama Papers' documents reveal widespread offshore tax avoidance by global political and business elites, triggering political fallout in multiple countries including the resignation of Iceland's prime minister."),
]

events = pd.DataFrame(EVENTS, columns=['date', 'category', 'actor', 'text'])
events['date'] = pd.to_datetime(events['date'])
events.insert(0, 'event_id', ['EVT_%03d' % (i+1) for i in range(len(events))])

# ── Join with real market data to compute labels ─────────────────────────────
fin = pd.read_csv(r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data\financial_data.csv',
                  parse_dates=['date'], index_col='date')

def forward_return(series_price, event_date, horizon_days):
    """Log return from last close at/before event to close at event+horizon."""
    idx = series_price.index
    base_pos = idx.searchsorted(event_date, side='right') - 1
    if base_pos < 0:
        return np.nan
    tgt_date = event_date + pd.Timedelta(days=horizon_days)
    tgt_pos = idx.searchsorted(tgt_date, side='right') - 1
    if tgt_pos <= base_pos or tgt_pos >= len(idx):
        return np.nan
    return float(np.log(series_price.iloc[tgt_pos] / series_price.iloc[base_pos]))

THRESH = 0.001  # 0.1% neutral band
def direction(r):
    if pd.isna(r): return np.nan
    if r >  THRESH: return 'UP'
    if r < -THRESH: return 'DOWN'
    return 'NEUTRAL'

for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
    for h in [1, 3, 7]:
        col = f'{pair}_ret_{h}d'
        events[col] = events['date'].apply(lambda d: forward_return(fin[pair], d, h))
        events[f'{pair}_dir_{h}d'] = events[col].apply(direction)

# VIX change (level change over 3 days)
events['VIX_chg_3d'] = events['date'].apply(lambda d: forward_return(fin['VIX'], d, 3))
events['VIX_dir_3d'] = events['VIX_chg_3d'].apply(direction)

out_path = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data\political_events.csv'
events.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f'Events: {len(events)}')
print(events['category'].value_counts().to_string())
print()
print('Label distribution EURUSD_dir_3d:')
print(events['EURUSD_dir_3d'].value_counts(dropna=False).to_string())
print()
print(f'Saved: {out_path}')
