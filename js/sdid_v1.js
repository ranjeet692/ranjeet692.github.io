/**
 * cross-storage - Cross domain local storage
 *
 * @version   1.0.0
 * @link      https://github.com/zendesk/cross-storage
 * @author    Daniel St. Jules <danielst.jules@gmail.com>
 * @copyright Zendesk
 * @license   Apache-2.0
 */

 !function(e) {
    function t(e, r) {
        r = r || {},
        this._id = t._generateUUID(),
        this._promise = r.promise || Promise,
        this._frameId = r.frameId || "CrossStorageClient-" + this._id,
        this._origin = t._getOrigin(e),
        this._requests = {},
        this._connected = !1,
        this._closed = !1,
        this._count = 0,
        this._timeout = r.timeout || 5e3,
        this._listener = null,
        this._installListener();
        var o;
        r.frameId && (o = document.getElementById(r.frameId)),
        o && this._poll(),
        o = o || this._createFrame(e),
        this._hub = o.contentWindow
    }
    t.frameStyle = {
        display: "none",
        position: "absolute",
        top: "-999px",
        left: "-999px"
    },
    t._getOrigin = function(e) {
        var t, r, o;
        return t = document.createElement("a"),
        t.href = e,
        t.host || (t = window.location),
        r = t.protocol && ":" !== t.protocol ? t.protocol : window.location.protocol,
        o = r + "//" + t.host,
        o = o.replace(/:80$|:443$/, "")
    }
    ,
    t._generateUUID = function() {
        return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(e) {
            var t = 16 * Math.random() | 0
              , r = "x" == e ? t : 3 & t | 8;
            return r.toString(16)
        })
    }
    ,
    t.prototype.onConnect = function() {
        var e = this;
        return this._connected ? this._promise.resolve() : this._closed ? this._promise.reject(new Error("CrossStorageClient has closed")) : (this._requests.connect || (this._requests.connect = []),
        new this._promise(function(t, r) {
            var o = setTimeout(function() {
                r(new Error("CrossStorageClient could not connect"))
            }, e._timeout);
            e._requests.connect.push(function(e) {
                return clearTimeout(o),
                e ? r(e) : (t(),
                void 0)
            })
        }
        ))
    }
    ,
    t.prototype.set = function(e, t) {
        return this._request("set", {
            key: e,
            value: t
        })
    }
    ,
    t.prototype.get = function() {
        var e = Array.prototype.slice.call(arguments);
        return this._request("get", {
            keys: e
        })
    }
    ,
    t.prototype.del = function() {
        var e = Array.prototype.slice.call(arguments);
        return this._request("del", {
            keys: e
        })
    }
    ,
    t.prototype.clear = function() {
        return this._request("clear")
    }
    ,
    t.prototype.getKeys = function() {
        return this._request("getKeys")
    }
    ,
    t.prototype.close = function() {
        var e = document.getElementById(this._frameId);
        e && e.parentNode.removeChild(e),
        window.removeEventListener ? window.removeEventListener("message", this._listener, !1) : window.detachEvent("onmessage", this._listener),
        this._connected = !1,
        this._closed = !0
    }
    ,
    t.prototype._installListener = function() {
        var e = this;
        this._listener = function(t) {
            var r, o, n, s;
            if (!e._closed && t.data && "string" == typeof t.data && (o = "null" === t.origin ? "file://" : t.origin,
            o === e._origin))
                if ("cross-storage:unavailable" !== t.data) {
                    if (-1 !== t.data.indexOf("cross-storage:") && !e._connected) {
                        if (e._connected = !0,
                        !e._requests.connect)
                            return;
                        for (r = 0; r < e._requests.connect.length; r++)
                            e._requests.connect[r](n);
                        delete e._requests.connect
                    }
                    if ("cross-storage:ready" !== t.data) {
                        try {
                            s = JSON.parse(t.data)
                        } catch (i) {
                            return
                        }
                        s.id && e._requests[s.id] && e._requests[s.id](s.error, s.result)
                    }
                } else {
                    if (e._closed || e.close(),
                    !e._requests.connect)
                        return;
                    for (n = new Error("Closing client. Could not access localStorage in hub."),
                    r = 0; r < e._requests.connect.length; r++)
                        e._requests.connect[r](n)
                }
        }
        ,
        window.addEventListener ? window.addEventListener("message", this._listener, !1) : window.attachEvent("onmessage", this._listener)
    }
    ,
    t.prototype._poll = function() {
        var e, t, r;
        e = this,
        r = "file://" === e._origin ? "*" : e._origin,
        t = setInterval(function() {
            return e._connected ? clearInterval(t) : (e._hub && e._hub.postMessage("cross-storage:poll", r),
            void 0)
        }, 1e3)
    }
    ,
    t.prototype._createFrame = function(e) {
        var r, o;
        r = window.document.createElement("iframe"),
        r.id = this._frameId;
        for (o in t.frameStyle)
            t.frameStyle.hasOwnProperty(o) && (r.style[o] = t.frameStyle[o]);
        return window.document.body.appendChild(r),
        r.src = e,
        r
    }
    ,
    t.prototype._request = function(e, t) {
        var r, o;
        return this._closed ? this._promise.reject(new Error("CrossStorageClient has closed")) : (o = this,
        o._count++,
        r = {
            id: this._id + ":" + o._count,
            method: "cross-storage:" + e,
            params: t
        },
        new this._promise(function(e, t) {
            var n, s, i;
            n = setTimeout(function() {
                o._requests[r.id] && (delete o._requests[r.id],
                t(new Error("Timeout: could not perform " + r.method)))
            }, o._timeout),
            o._requests[r.id] = function(s, i) {
                return clearTimeout(n),
                delete o._requests[r.id],
                s ? t(new Error(s)) : (e(i),
                void 0)
            }
            ,
            Array.prototype.toJSON && (s = Array.prototype.toJSON,
            Array.prototype.toJSON = null),
            i = "file://" === o._origin ? "*" : o._origin,
            o._hub.postMessage(JSON.stringify(r), i),
            s && (Array.prototype.toJSON = s)
        }
        ))
    }
    ,
    "undefined" != typeof module && module.exports ? module.exports = t : "undefined" != typeof exports ? exports.CrossStorageClient = t : "function" == typeof define && define.amd ? define([], function() {
        return t
    }) : e.CrossStorageClient = t
}(this);
