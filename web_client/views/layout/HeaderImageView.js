import { restRequest } from 'girder/rest';

import events from '../../events';
import View from '../View';

import headerImageTemplate from '../../templates/layout/headerImage.pug';
import '../../stylesheets/layout/headerImage.styl';

var HeaderImageView = View.extend({
    events: {
        'click .h-open-image': function (evt) {
            events.trigger('h:openImageUi');
        },
        'click .h-open-annotated-image': function (evt) {
            events.trigger('h:openAnnotatedImageUi');
        }
    },

    initialize() {
        this.imageModel = null;
        this.parentChain = null;
        this.listenTo(events, 'h:imageOpened', (model) => {
            this.imageModel = model;
            this.parentChain = null;
            this.nextImageLink = null;
            this.previousImageLink = null;
            if (model) {
                $.when(
                    restRequest({
                        url: `item/${model.id}/previous_image`
                    }).done((previous) => {
                        if (previous._id !== model.id) {
                            this.previousImageLink = `#?image=${previous._id}`;
                        }
                    }),
                    restRequest({
                        url: `item/${model.id}/next_image`
                    }).done((next) => {
                        if (next._id !== model.id) {
                            this.nextImageLink = `#?image=${next._id}`;
                        }
                    }),
                    this.imageModel.getRootPath((resp) => {
                        this.parentChain = resp;
                    })
                ).done(() => this.render());
            }
            this.render();
        });
    },

    render() {
        this.$el.html(headerImageTemplate({
            image: this.imageModel,
            parentChain: this.parentChain,
            nextImageLink: this.nextImageLink,
            previousImageLink: this.previousImageLink
        }));
        return this;
    }
});

export default HeaderImageView;
