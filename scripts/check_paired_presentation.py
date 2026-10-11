"""Run browser acceptance checks; requires Playwright Chromium and pypdf."""
from pathlib import Path
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
import os
os.chdir(Path(__file__).resolve().parents[1])
Path('outputs').mkdir(exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch()
 page=browser.new_page(viewport={'width':1440,'height':1000})
 errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(Path('docs/reports/Paired-study-2026-10-10-presentation.html').resolve().as_uri())
 evidence=page.evaluate('data')
 assert len(evidence['selected_comparisons'])==6
 for row in evidence['selected_comparisons']:
  assert f"{row['new_task_b']:.1f}%" in page.locator('#new-chart').inner_text()
  expected=('+' if row['retained_mean_difference']>0 else '')+f"{row['retained_mean_difference']:.2f}"
  assert expected in page.locator('#retained-chart').inner_text()
 assert page.locator('#new-chart rect').count()==9
 assert page.locator('#forget-chart polyline').count()==6
 assert page.locator('#retained-chart rect').count()==6
 print('Slides' ,page.locator('.slide').count(),'experiments',page.locator('.experiment').count())
 page.screenshot(path='outputs/presentation-slide1.png',full_page=True)
 page.click('#next');page.screenshot(path='outputs/presentation-slide2.png',full_page=True)
 page.click('#next');page.screenshot(path='outputs/presentation-slide3.png',full_page=True)
 page.click('#explore-tab')
 for i in range(9):
  b=page.locator('.experiment').nth(i);id=b.get_attribute('data-id');b.click()
  assert page.locator('#details code').first.inner_text()==id
  assert page.locator('#tree-branches button.selected').get_attribute('data-id')==id
  assert page.locator('#tree-source button').count()==8
  assert page.locator('#tree-branches button').count()==3
  if 'branches/' in id: assert page.locator('#details tbody tr').count()==8
  else: assert page.locator('#details table').count()==0
 page.locator('#tree-source button').nth(2).click()
 assert 'position 3 of 8' in page.locator('#source-info').inner_text()
 page.locator('#tree-branches button').first.click()
 assert 'controls/' in page.locator('#details code').first.inner_text()
 page.screenshot(path='outputs/presentation-tree.png',full_page=True)
 page.click('#presentation-tab');page.locator('body').click(position={'x':5,'y':150});page.keyboard.press('ArrowLeft')
 assert 'Slide 2 / 4' in page.locator('#slide-count').inner_text()
 page.click('#notes-toggle');assert page.locator('#notes').is_visible()
 page.locator('#presentation-tab').focus();page.keyboard.press('ArrowRight');assert page.locator('#explore-panel').is_visible()
 page.click('#fullscreen');page.wait_for_function('!!document.fullscreenElement')
 page.click('#fullscreen');page.wait_for_function('!document.fullscreenElement')
 page.emulate_media(media='print')
 assert page.locator('.slide:visible').count()==4
 page.pdf(path='outputs/presentation-print-check.pdf',prefer_css_page_size=True,print_background=True)
 assert len(PdfReader('outputs/presentation-print-check.pdf').pages)==4
 assert not errors,errors
 print('All nine selections, source nodes, branch nodes, tabs, keyboard, notes, fullscreen and print visibility passed.')
 browser.close()
