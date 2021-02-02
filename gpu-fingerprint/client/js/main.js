
jQuery(function ($) {
	//Preloader
	var preloader = $('.preloader');
	$(window).load(function () {
		preloader.remove();
		$("#fingerprint-status").html("Running");
		$("#fingerprint-iframe").attr("src", "./js/fingerprint/index.html");
	});
});