var client = new ClientJS();
		var fingerprint = client.getFingerprint();
		window.document.getElementById('fingerprint').innerHTML = fingerprint;
		var browser = client.getBrowserData();
		var device = client.getDevice();
		var cpu = client.getCPU();
		window.document.getElementById('ua').innerHTML = browser.ua;
		window.document.getElementById('browser').innerHTML = browser.browser.name + '(' + browser.browser.version + ')';
		window.document.getElementById('os').innerHTML = browser.os.name + '(' + browser.os.version + ')';
		window.document.getElementById('device').innerHTML = JSON.stringify(device);
		window.document.getElementById('engine').innerHTML = browser.engine.name + '(' + browser.engine.version + ')';
		window.document.getElementById('cpu').innerHTML = browser.cpu.architecture + ' / ' + JSON.stringify(cpu);
		var sp = client.getScreenPrint();
		var cd = client.getColorDepth();
		var cr = client.getCurrentResolution();
		var ar = client.getAvailableResolution();
		var dxdpi = client.getDeviceXDPI();
		var dydpi = client.getDeviceYDPI();
		window.document.getElementById('sp').innerHTML = sp;
		window.document.getElementById('cd').innerHTML = cd;
		window.document.getElementById('cr').innerHTML = cr;
		window.document.getElementById('ar').innerHTML = ar;
		window.document.getElementById('dxdpi').innerHTML = dxdpi;
		window.document.getElementById('dydpi').innerHTML = dydpi;

		var timezone = client.getTimeZone();
		var language = client.getLanguage();
		var systemLanguage = client.getSystemLanguage();
		var isCanvas = client.isCanvas();
		var canvasPrint = client.getCanvasPrint();

		window.document.getElementById('timezone').innerHTML = timezone;
		window.document.getElementById('language').innerHTML = language;
		window.document.getElementById('systemLanguage').innerHTML = systemLanguage;
		window.document.getElementById('isCanvas').innerHTML = isCanvas;
		window.document.getElementById('canvasPrint').innerHTML = canvasPrint;

		var isFont = client.isFont();
		var font = client.getFonts();
		window.document.getElementById('font').innerHTML = isFont + ' / ' + font;

		var plugin = client.getPlugins();
		window.document.getElementById('plugin').innerHTML = plugin;

		function clipboardCopy() {
			var copyText = window.document.getElementById('canvasPrint');
			copyText.select();
			copyText.setSelectionRange(0, 99999); /* For mobile devices */
			window.document.execCommand("copy");
			window.document.getElementById('copy').innerHTML = 'Copied!';
		}

		function compareCanvas() {
			var canvas_a = window.document.getElementById('canvasPrint').value;
			var canvas_b = window.document.getElementById('canvasPrintCompare').value;
			var match = canvas_a.localeCompare(canvas_b);
			if (match === 0) {
				alert('These canvas are identical!');
			} else {
				alert('These canvas are not identical!');
			}
		}