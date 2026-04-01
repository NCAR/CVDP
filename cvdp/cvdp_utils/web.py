def generate_webpages(config_dict):
    # Website Creation
    from jinja2 import Template
    #Make jinja functions that mimics python functions.
    #  - This will allow for the use of 'list' in the html rendering.
    def jinja_list(seas_list):
        return list(seas_list)
    #   - This will allow for the use of 'enumerate' in the html rendering.
    def jinja_enumerate(arg):
        return enumerate(arg)

    import cvdp_utils.utils as utils
    season_list = utils.season_list # = ["DJF","JFM","MAM","JJA","JAS","SON","ANN"]
    var_seasons = utils.var_seasons

    import os, time
    from datetime import datetime, timezone

    plot_loc = config_dict["plot_loc"]
    def img(plot_file):
        path = plot_loc / plot_file
        return plot_file if os.path.exists(path) else "file_not_found_image.png"

    # Get the current time as a timezone-aware datetime object in UTC
    utc_time_aware = datetime.now(timezone.utc)

    # Get the UTC timestamp (seconds since the Unix epoch)
    utc_timestamp = utc_time_aware.timestamp() # or simply time.time()

    print(f"UTC Time (aware): {utc_time_aware}")
    print(f"UTC Timestamp: {utc_timestamp}")

    tmp_rnd_dict = {"title":"Diagnostics Plots",
                    "img":lambda f: img(f),
                    "create_time":utc_time_aware
                    }

    def _make_html(template, html_out, tmp_rnd_dict={}):
        """
        Make html script from Jinja
        """
        # Individual main CVDP index html
        with open(template) as f:
            template = Template(f.read())
        # render html
        html = template.render(tmp_rnd_dict)
        # write output
        with open(html_out, "w") as f:
            f.write(html)

    _make_html("template_main.html", "index.html", tmp_rnd_dict)
    # Individual members index html file if applicable
    _make_html("template_indmem.html", "index_indmem.html", tmp_rnd_dict)

    # Comaprison to NCL Plots if applicable
    if "ncl_plot_loc" in config_dict:
        print("\tComapring to premade NCL plots for comparison...")
        ncl_plot_loc = config_dict["ncl_plot_loc"]
        tmp_rnd_dict["ncl_plot_loc"] = ncl_plot_loc
        print(f"\tncl_plot_loc: {ncl_plot_loc}")

        _make_html("template_ncl_compare.html", "ncl_compare_index.html", tmp_rnd_dict)
        # Compare individual members index html file if applicable
        _make_html("template_ncl_compare_indmem.html", "ncl_compare_indmem_index.html", tmp_rnd_dict)
    else:
        print("\tNo NCL plot comparison I guess...")
    print("All done yay and stuff. Check your inbox there is some activity....   ...You've got mail!")