var h=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{};function _(t){if(t.__esModule)return t;var e=t.default;if(typeof e=="function"){var r=function n(){if(this instanceof n){var o=[null];o.push.apply(o,arguments);var f=Function.bind.apply(e,o);return new f}return e.apply(this,arguments)};r.prototype=e.prototype}else r={};return Object.defineProperty(r,"__esModule",{value:!0}),Object.keys(t).forEach(function(n){var o=Object.getOwnPropertyDescriptor(t,n);Object.defineProperty(r,n,o.get?o:{enumerable:!0,get:function(){return t[n]}})}),r}function p(){return p=Object.assign?Object.assign.bind():function(t){for(var e=1;e<arguments.length;e++){var r=arguments[e];for(var n in r)Object.prototype.hasOwnProperty.call(r,n)&&(t[n]=r[n])}return t},p.apply(this,arguments)}const s=Object.freeze(Object.defineProperty({__proto__:null,default:p},Symbol.toStringTag,{value:"Module"}));function v(t){if(t===void 0)throw new ReferenceError("this hasn't been initialised - super() hasn't been called");return t}function i(t,e){return i=Object.setPrototypeOf?Object.setPrototypeOf.bind():function(n,o){return n.__proto__=o,n},i(t,e)}function w(t,e){t.prototype=Object.create(e.prototype),t.prototype.constructor=t,i(t,e)}const j=_(s);function O(){if(typeof Reflect>"u"||!Reflect.construct||Reflect.construct.sham)return!1;if(typeof Proxy=="function")return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],function(){})),!0}catch{return!1}}function u(t,e,r){return O()?u=Reflect.construct.bind():u=function(o,f,l){var c=[null];c.push.apply(c,f);var b=Function.bind.apply(o,c),y=new b;return l&&i(y,l.prototype),y},u.apply(null,arguments)}function m(t,e){if(typeof e!="function"&&e!==null)throw new TypeError("Super expression must either be null or a function");t.prototype=Object.create(e&&e.prototype,{constructor:{value:t,writable:!0,configurable:!0}}),Object.defineProperty(t,"prototype",{writable:!1}),e&&i(t,e)}function a(t){return a=Object.setPrototypeOf?Object.getPrototypeOf.bind():function(r){return r.__proto__||Object.getPrototypeOf(r)},a(t)}function g(t){return Function.toString.call(t).indexOf("[native code]")!==-1}function d(t){var e=typeof Map=="function"?new Map:void 0;return d=function(n){if(n===null||!g(n))return n;if(typeof n!="function")throw new TypeError("Super expression must either be null or a function");if(typeof e<"u"){if(e.has(n))return e.get(n);e.set(n,o)}function o(){return u(n,arguments,a(this).constructor)}return o.prototype=Object.create(n.prototype,{constructor:{value:o,enumerable:!1,writable:!0,configurable:!0}}),i(o,n)},d(t)}export{w as _,v as a,p as b,h as c,m as d,d as e,u as f,_ as g,j as r};
