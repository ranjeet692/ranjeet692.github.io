(function (scope) {
	'use strict';

	var SdidJS = function(url) {
		this._mid = '';
		this._bid = '';
		return this;
	};

	SdidJS.prototype = {
		generate: function() {
			//checkLocalStorage
			if (localStorage.getItem('mid')) {
				this._mid = localStorage.getItem('mid');
				this._bid = localStorage.getItem('bid');
			} else {
				this.forceGenerate();
			}
		},

		forceGenerate: function() {
			var _cframe = document.createElement("iframe");
			_cframe.setAttribute('width', "0");
			_cframe.setAttribute('height', "0");
			_cframe.setAttribute('src', "dist/index.html");
			_cframe.setAttribute('frameborder', "0");
			_cframe.setAttribute('scrolling', 'no');
			_cframe.setAttribute('style', 'width:0; height:0; border:0; border:none;');
			document.body.appendChild(_cframe);
		},

		getMid: function() {
			if (!this._mid) {
				this.generate();
			}
			return this._mid;
		},

		getBid: function() {
			if (!this._bid) {
				this.generate();	
			}
			return this._bid;
		}
	};

	if (typeof module === 'object' && typeof exports !== "undefined") {
		module.exports = SdidJS;
	}
	scope.SdidJS = SdidJS;

})(window);  