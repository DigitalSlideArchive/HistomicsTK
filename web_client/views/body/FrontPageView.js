import { cancelRestRequest, apiRoot, staticRoot } from 'girder/rest';
import { getCurrentUser } from 'girder/auth';
import * as version from 'girder/version';
import GirderFrontPageView from 'girder/views/FrontPageView';

import frontPageTemplate from '../../templates/body/frontPage.pug';
import '../../stylesheets/body/frontPage.styl';

var FrontPageView = GirderFrontPageView.extend({
    events: {},

    initialize: function () {
        cancelRestRequest('fetch');
        this.render();
    },

    render: function () {
        this.$el.addClass('h-body-nopad');

        this.$el.html(frontPageTemplate({
            apiRoot,
            staticRoot,
            version,
            currentUser: getCurrentUser()
        }));
    }
});

export default FrontPageView;
