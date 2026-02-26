"""
Browser Automation Tool (Playwright Integration)

Provides a persistent browser session for the Agent to inspect and interact with web pages.
"""

import os
import time
import base64
from typing import Dict, Optional, List, Any

try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext, Route
except ImportError:
    sync_playwright = None
    Page = None
    BrowserContext = None

class BrowserManager:
    """
    Manages a persistent Playwright browser session.
    """
    
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None
        self.page: Optional[Page] = None
        self._ensure_playwright_installed()

    def _ensure_playwright_installed(self):
        if sync_playwright is None:
            print("[Warning] Playwright not installed. Browser tools will be unavailable.")

    def start(self):
        """Start the browser session if not already running."""
        global sync_playwright, Page, BrowserContext
        
        if self.page:
            return

        if not sync_playwright:
            # 动态尝试再次导入（支持 Agent 运行时动态 pip install 的场景）
            try:
                from playwright.sync_api import sync_playwright as sp
                sync_playwright = sp
            except ImportError:
                raise ImportError("Playwright is not installed. Please run `pip install playwright` and `playwright install`.")

        try:
            self.playwright = sync_playwright().start()
            # Launch persistent context or just a browser? 
            # We use a regular launch to start.
            self.browser = self.playwright.chromium.launch(headless=self.headless)
            self.context = self.browser.new_context(
                 viewport={"width": 1280, "height": 800},
                 user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            self.page = self.context.new_page()
            print("[System] Browser started.")
        except Exception as e:
            print(f"[Error] Failed to start browser: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the browser session."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        
        self.context = None
        self.browser = None
        self.playwright = None
        self.page = None

    def ensure_page(self):
        if not self.page:
            self.start()

    def goto(self, url: str) -> str:
        """Navigate to a URL."""
        self.ensure_page()
        try:
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            status = response.status if response else "unknown"
            return f"<success>Navigated to {url} (Status: {status})</success>"
        except Exception as e:
            return f"<error>Navigation failed: {str(e)}</error>"

    def get_content(self, selector: str = None) -> str:
        """
        Get page content.
        If selector is provided, return text of that element.
        Otherwise, return a simplified summary of the page structure.
        """
        self.ensure_page()
        try:
            if selector:
                if not self.page.locator(selector).count():
                     return f"<error>Selector '{selector}' not found</error>"
                return self.page.locator(selector).inner_text()
            else:
                title = self.page.title()
                # Simple text dump
                text = self.page.inner_text("body")
                preview = text[:2000] + "..." if len(text) > 2000 else text
                
                # Fetch interactive elements for LLM to know how to select
                interactives = self.page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a, button, input, textarea, select'))
                        .filter(el => {
                            let rect = el.getBoundingClientRect();
                            return rect.width > 0 && rect.height > 0;
                        })
                        .map(el => {
                            let text = (el.innerText || el.value || el.placeholder || '').trim().replace(/\\n/g, ' ');
                            let info = `<${el.tagName.toLowerCase()}`;
                            if(el.id) info += ` id="${el.id}"`;
                            if(el.name) info += ` name="${el.name}"`;
                            if(el.className && typeof el.className === 'string') info += ` class="${el.className}"`;
                            info += `>`;
                            if(text) info += text.substring(0, 50);
                            return info;
                        }).slice(0, 100); // 限制前100个元素防止过长
                }''')
                
                interactives_text = "\n".join(interactives)
                
                return f"Page: {title} ({self.page.url})\n\n[Interactive Elements (Available for `browser_act`)]:\n{interactives_text}\n\n[Content Preview]:\n{preview}"
                
        except Exception as e:
            return f"<error>Get content failed: {str(e)}</error>"

    def act(self, action: str, selector: str, value: str = None) -> str:
        """Perform an action on the page."""
        self.ensure_page()
        try:
            target = self.page.locator(selector)
            if not target.count():
                 return f"<error>Element '{selector}' not found</error>"

            if action == "click":
                target.first.click()
                return f"<success>Clicked {selector}</success>"
            elif action == "type":
                if value is None:
                    return "<error>Value required for type action</error>"
                target.first.fill(value)
                return f"<success>Typed '{value}' into {selector}</success>"
            elif action == "hover":
                target.first.hover()
                return f"<success>Hovered over {selector}</success>"
            else:
                return f"<error>Unknown action: {action}</error>"
                
        except Exception as e:
            return f"<error>Action failed: {str(e)}</error>"

    def screenshot(self) -> str:
        """Take a screenshot and return the path."""
        self.ensure_page()
        try:
            # Save to a temp file in current working dir
            filename = f"screenshot_{int(time.time())}.png"
            path = os.path.abspath(filename)
            self.page.screenshot(path=path)
            return f"<success>Screenshot saved to {path}</success>"
        except Exception as e:
            return f"<error>Screenshot failed: {str(e)}</error>"

    def get_console_logs(self) -> str:
        # Implementing log capture requires event listeners.
        # For simplicity, we might skip this or implement a log buffer.
        return "<info>Console logs not implemented yet</info>"
