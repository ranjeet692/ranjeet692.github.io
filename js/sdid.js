!function(d, w){
"use strict";

var sdk = {
    getId : function(callback) {
        var name = 'SummitClientId';
        var hub = 'https://idstatic.summitmedia-digital.com/project/user-session-unification/1.0/html/hub.html';
        var cid = '';
            var crypto = self.crypto || self.msCrypto;
            if (crypto) {
                var csize = 24;
                var calpha = '0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz';
                var cbytes = crypto.getRandomValues(new Uint8Array(csize));
                while (0 < csize--) {
                    cid += calpha[cbytes[csize] & 63];
                }
		    console.log('scid :: ' + cid);
            }
            return cid || id;
    }
}

var summitclientid = {
    id1: null,
    get: function (callback, retry=0) {
        if (summitclientid.id1 !== null) {
            callback(summitclientid.id1);
        } else {
            retry++;
            if (retry > 0 && retry <= 15) {
                setTimeout(function (){ summitclientid.get(callback, retry); }, retry * 500);
            }
        }
    }
}

sdk.getId(function (id) {
   summitclientid.id1 = id;  
});

if (typeof w.summitclientid === 'undefined') {
    w.summitclientid = summitclientid;
}

}(document, window);

