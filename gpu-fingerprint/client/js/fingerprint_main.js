(function (scope) {
	'use strict';

	var SdidJS = function(url) {
		this._backendHost = url || '127.0.0.1:5000';
		this._mid = '';
		this._bid = '';
		return this;
	};

	SdidJS.prototype = {
		generate: function() {
			//checkLocalStorage
			this.forceGenerate();
		},

		forceGenerate: function() {
			var _cframe = document.createElement("iframe");
			_cframe.setAttribute('width', "0");
			_cframe.setAttribute('height', "0");
			_cframe.setAttribute('src', "http://" + this._backendHost + "/");
			_cframe.setAttribute('frameborder', "0");
			_cframe.setAttribute('scrolling', 'no');
			_cframe.setAttribute('style', 'width:0; height:0; border:0; border:none;');
			document.body.appendChild(_cframe);
		},

		getMid: function() {
			if (!_mid) {
				this.generate();
			}
			return _mid;
		},

		getBid: function() {
			if (!_bid) {
				this.generate();	
			}
			return _bid;
		}
	};

	if (typeof module === 'object' && typeof exports !== "undefined") {
		module.exports = SdidJS;
	}
	scope.SdidJS = SdidJS;

})(window);  